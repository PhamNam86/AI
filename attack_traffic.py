import switch
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from datetime import datetime
import os

class CollectTrainingStatsApp(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(CollectTrainingStatsApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)
        self.file_name = "FlowStatsfile.csv"

        # Khởi tạo tiêu đề file nếu file chưa tồn tại
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w") as f:
                f.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,'
                        'icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,'
                        'hard_timeout,flags,packet_count,byte_count,packet_count_per_second,'
                        'byte_count_per_second,label\n')

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(10)

    def request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        with open(self.file_name, "a+") as file0:
            # Lọc các flow có priority 1 và là IPv4 để tránh lỗi match
            for stat in body:
                if stat.priority != 1:
                    continue
                
                # Sử dụng get() để tránh lỗi KeyError
                ip_src = stat.match.get('ipv4_src')
                ip_dst = stat.match.get('ipv4_dst')
                ip_proto = stat.match.get('ip_proto')

                if not ip_src: # Bỏ qua nếu không phải luồng IPv4
                    continue

                icmp_code, icmp_type = -1, -1
                tp_src, tp_dst = 0, 0

                if ip_proto == 1: # ICMP
                    icmp_code = stat.match.get('icmpv4_code', -1)
                    icmp_type = stat.match.get('icmpv4_type', -1)
                elif ip_proto == 6: # TCP
                    tp_src = stat.match.get('tcp_src', 0)
                    tp_dst = stat.match.get('tcp_dst', 0)
                elif ip_proto == 17: # UDP
                    tp_src = stat.match.get('udp_src', 0)
                    tp_dst = stat.match.get('udp_dst', 0)

                flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
                
                # Tính toán tốc độ gói tin (tránh chia cho 0)
                duration = stat.duration_sec + (stat.duration_nsec / 10**9)
                packet_count_per_second = stat.packet_count / duration if duration > 0 else 0
                byte_count_per_second = stat.byte_count / duration if duration > 0 else 0

                # Label: 1 cho Attack, 0 cho Normal
                label = 1 

                file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                    .format(timestamp, dpid, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                            ip_proto, icmp_code, icmp_type,
                            stat.duration_sec, stat.duration_nsec,
                            stat.idle_timeout, stat.hard_timeout,
                            stat.flags, stat.packet_count, stat.byte_count,
                            packet_count_per_second, byte_count_per_second, label))