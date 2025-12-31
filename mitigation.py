from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
import pandas as pd
import numpy as np
import socket
import struct
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import switchm

class SimpleMonitor13(switchm.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.flow_model = None
        self.scaler = StandardScaler()
        
        # 1. ĐỒNG BỘ THỨ TỰ CỘT (Sửa lại danh sách này cho khớp 100% với RF_random-forest.py của bạn)
        self.feature_columns = [
    'ip_src', 'tp_src', 'ip_dst', 'tp_dst', 'ip_proto', 
    'icmp_code', 'icmp_type', 'flow_duration_sec', 
    'packet_count', 'byte_count', 'pps'  # Thêm pps vào cuối
]
        
        self.flow_training()

    # --- HÀM CHUYỂN IP SANG SỐ (Yêu cầu nâng cao) ---
    def ip_to_int(self, ip):
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except:
            return 0

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            if self.flow_model is not None:
                self.flow_predict()

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        filename = "PredictFlowStatsfile.csv"
        #self.logger.info(">>> Nhận dữ liệu từ Switch %016x", ev.msg.datapath.id)
        
        with open(filename, "w") as f:
            # Ghi tiêu đề cột bao gồm 'pps'
            f.write(','.join(self.feature_columns) + '\n')
            
            for stat in body:
                # Chỉ xử lý các luồng có priority=1 như trong dump-flows
                if stat.priority == 1:
                    match = stat.match
                    
                    # 1. Trích xuất và chuyển đổi IP sang số
                    ip_src = self.ip_to_int(match.get('ipv4_src', '0.0.0.0'))
                    ip_dst = self.ip_to_int(match.get('ipv4_dst', '0.0.0.0'))
                    
                    # 2. Trích xuất các trường giao thức và cổng
                    ip_proto = match.get('ip_proto', 0)
                    tp_src = match.get('tcp_src', match.get('udp_src', 0))
                    tp_dst = match.get('tcp_dst', match.get('udp_dst', 0))
                    icmp_code = match.get('icmpv4_code', -1)
                    icmp_type = match.get('icmpv4_type', -1)
                    
                    # 3. Tính toán đặc trưng PPS (Packets Per Second) nâng cao
                    # Tránh chia cho 0 bằng cách lấy tối thiểu là 1 giây
                    duration = stat.duration_sec if stat.duration_sec > 0 else 1
                    pps = stat.packet_count / duration

                    # 4. Ghi dữ liệu theo đúng thứ tự cột khai báo
                    # Đảm bảo sử dụng dấu phẩy chính xác giữa các trường
                    line = (f"{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},"
                            f"{icmp_code},{icmp_type},{stat.duration_sec},"
                            f"{stat.packet_count},{stat.byte_count},{pps}\n")
                    f.write(line)
        
    #    self.logger.info(">>> ĐÃ CẬP NHẬT DỮ LIỆU THÀNH CÔNG VÀO %s", filename)

    # --- TIỀN XỬ LÝ NÂNG CAO & AN TOÀN KHI THIẾU FEATURE ---
    def preprocess_data(self, df, is_training=False):
        # 1. Đảm bảo dữ liệu chỉ lấy các cột cần thiết và đúng thứ tự
        # Nếu thiếu feature, code này sẽ tự bổ sung cột rỗng (An toàn)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_columns].copy()
        X.fillna(0, inplace=True)
        
        # 2. Xử lý IP nếu dữ liệu huấn luyện vẫn là dạng chuỗi
        if is_training:
            for col in ['ip_src', 'ip_dst']:
                if X[col].dtype == object:
                    X[col] = X[col].apply(self.ip_to_int)
        
        if is_training:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def flow_training(self):
        """ HUẤN LUYỆN VÀ ĐÁNH GIÁ ACCURACY """
        if not os.path.exists('FlowStatsfile.csv'):
            self.logger.error("Lỗi: Thiếu dữ liệu huấn luyện!")
            return

        df = pd.read_csv('FlowStatsfile.csv')
        X = self.preprocess_data(df, is_training=True)
        y = df.iloc[:, -1].values # Giả định cột cuối là nhãn

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Thông số mô hình nên đồng bộ với RF_random-forest.py
        self.flow_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.flow_model.fit(X_train, y_train)
        
        # 3. ĐÁNH GIÁ ACCURACY
        y_pred = self.flow_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.logger.info("--- HUẤN LUYỆN XONG - ĐỘ CHÍNH XÁC (Accuracy): %.2f%% ---", acc * 100)

    def flow_predict(self):
        try:
            if not os.path.exists('PredictFlowStatsfile.csv'): return
            predict_df = pd.read_csv('PredictFlowStatsfile.csv')
            if predict_df.empty: return

            X_predict = self.preprocess_data(predict_df, is_training=False)
            y_flow_pred = self.flow_model.predict(X_predict)

            # Thay vì chặn ngay khi có 1 luồng, hãy tính tỷ lệ
            ddos_count = np.count_nonzero(y_flow_pred == 1)
            total_flows = len(y_flow_pred)
            
            # Tính tỷ lệ phần trăm luồng bị nghi ngờ
            ddos_ratio = (ddos_count / total_flows) if total_flows > 0 else 0

            if ddos_ratio > 0.8:  # Chỉ chặn nếu > 80% luồng trong mạng là bất thường
                self.logger.warning("!!! CẢNH BÁO: XÁC NHẬN TẤN CÔNG DDOS (%d%% luồng) !!!", ddos_ratio * 100)
                self.mitigation = 1
            else:
                self.logger.info("Giám sát: Mạng bình thường (Nghi vấn: %d%%)", ddos_ratio * 100)
                self.mitigation = 0
                
        except Exception as e:
            self.logger.error("Lỗi dự đoán: %s", str(e))