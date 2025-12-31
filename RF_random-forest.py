from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from datetime import datetime
import pandas as pd
import numpy as np
import os

# Import các thư viện Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Chú ý: Đảm bảo file switchm.py nằm cùng thư mục
try:
    import switchm as switch 
except ImportError:
    from ryu.app import simple_switch_13 as switch

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        
        self.flow_model = None
        self.scaler = StandardScaler()
        
        # Danh sách 10 đặc trưng cơ bản (Bỏ pks_sec nếu file train chưa có)
        # Nếu file FlowStatsfile.csv của bạn có pks_sec, hãy thêm lại vào list này
        self.feature_columns = [
            'ip_src', 'tp_src', 'ip_dst', 'tp_dst', 'ip_proto', 
            'icmp_code', 'icmp_type', 'flow_duration_sec', 
            'packet_count', 'byte_count'
        ]
        
        self.flow_training_process()

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
                self.flow_predict_process()

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """ Thu thập dữ liệu thực tế """
        timestamp = datetime.now().timestamp()
        body = ev.msg.body
        filename = "PredictFlowStatsfile.csv"
        
        with open(filename, "w") as f:
            # Ghi header khớp với danh sách đặc trưng
            f.write('timestamp,datapath_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,packet_count,byte_count\n')
            
            for stat in body:
                if stat.priority == 1: 
                    ip_src = stat.match.get('ipv4_src', '0.0.0.0')
                    ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')
                    ip_proto = stat.match.get('ip_proto', 0)
                    tp_src = stat.match.get('tcp_src', stat.match.get('udp_src', 0))
                    tp_dst = stat.match.get('tcp_dst', stat.match.get('udp_dst', 0))
                    icmp_code = stat.match.get('icmpv4_code', -1)
                    icmp_type = stat.match.get('icmpv4_type', -1)

                    f.write(f"{timestamp},{ev.msg.datapath.id},{ip_src},{tp_src},{ip_dst},{tp_dst},"
                            f"{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},"
                            f"{stat.packet_count},{stat.byte_count}\n")

    def preprocess_data(self, df):
        """ Tiền xử lý dữ liệu an toàn """
        df_temp = df.copy()
        
        # Chuyển đổi IP
        for col in ['ip_src', 'ip_dst']:
            if col in df_temp.columns:
                df_temp[col] = df_temp[col].astype(str).str.replace('.', '', regex=False).astype(float)
        
        # Chỉ lấy các cột tồn tại trong cả file và danh sách mong muốn
        existing_features = [c for c in self.feature_columns if c in df_temp.columns]
        df_final = df_temp[existing_features].copy()
        
        # Nếu thiếu cột nào thì bù bằng 0
        for col in self.feature_columns:
            if col not in df_final.columns:
                df_final[col] = 0
        
        # Đảm bảo thứ tự cột luôn cố định
        df_final = df_final[self.feature_columns]
        df_final.fillna(0, inplace=True)
        return df_final

    def flow_training_process(self):
        self.logger.info("--- BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN ML ---")
        if not os.path.exists('FlowStatsfile.csv'):
            self.logger.error("LỖI: Không tìm thấy file FlowStatsfile.csv!")
            return

        try:
            dataset = pd.read_csv('FlowStatsfile.csv')
            y = dataset.iloc[:, -1].values 
            
            X_df = self.preprocess_data(dataset)
            X = X_df.values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
            
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            self.flow_model = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
            self.flow_model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, self.flow_model.predict(X_test))
            self.logger.info("HUẤN LUYỆN THÀNH CÔNG! Accuracy: %.2f%%", acc * 100)
            
        except Exception as e:
            self.logger.error("Lỗi huấn luyện: %s", str(e))

    def flow_predict_process(self):
        try:
            if not os.path.exists('PredictFlowStatsfile.csv'):
                return
                
            predict_df = pd.read_csv('PredictFlowStatsfile.csv')
            if predict_df.empty or len(predict_df) < 1:
                return

            victim_ips = predict_df['ip_dst'].values
            X_predict_df = self.preprocess_data(predict_df)
            X_predict = self.scaler.transform(X_predict_df.values)
            
            y_flow_pred = self.flow_model.predict(X_predict)
            
            ddos_count = np.count_nonzero(y_flow_pred == 1)

            if ddos_count > 0:
                self.logger.warning("!!! PHÁT HIỆN TẤN CÔNG DDOS !!!")
                idx = np.where(y_flow_pred == 1)[0][0]
                self.logger.warning("Nạn nhân: %s", victim_ips[idx])
            else:
                self.logger.info("Giám sát: Mạng An toàn")

        except Exception as e:
            self.logger.error("LỖI DỰ ĐOÁN: %s", str(e))