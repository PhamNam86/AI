from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, arp, ethernet, ether_types, ipv4, icmp, tcp, udp, in_proto

FLOW_SERIAL_NO = 0

def get_flow_number():
    global FLOW_SERIAL_NO
    FLOW_SERIAL_NO += 1
    return FLOW_SERIAL_NO

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.mitigation = 0  # Đặt là 1 nếu muốn bật tính năng tự động chặn cổng khi thấy IP lạ
        self.arp_ip_to_port = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, get_flow_number())

    def add_flow(self, datapath, priority, match, actions, serial_no, buffer_id=None, idle=0, hard=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        mod = parser.OFPFlowMod(datapath=datapath, cookie=serial_no, 
                                priority=priority, idle_timeout=idle, 
                                hard_timeout=hard, match=match, instructions=inst)
        if buffer_id:
            mod.buffer_id = buffer_id
        datapath.send_msg(mod)

    def block_port(self, datapath, portnumber):
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        actions = [] # Không có action = Drop (chặn)
        self.add_flow(datapath, 100, match, actions, get_flow_number(), hard=120)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.arp_ip_to_port.setdefault(dpid, {})
        self.arp_ip_to_port[dpid].setdefault(in_port, [])

        self.mac_to_port[dpid][eth.src] = in_port

        # Học địa chỉ IP từ ARP
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            a = pkt.get_protocol(arp.arp)
            if a.opcode in [arp.ARP_REQUEST, arp.ARP_REPLY]:
                if a.src_ip not in self.arp_ip_to_port[dpid][in_port]:
                    self.arp_ip_to_port[dpid][in_port].append(a.src_ip)

        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip_pkt = pkt.get_protocol(ipv4.ipv4)
                srcip = ip_pkt.src
                dstip = ip_pkt.dst
                protocol = ip_pkt.proto
                
                match = None

                # Tạo Match chi tiết theo Giao thức
                if protocol == in_proto.IPPROTO_ICMP:
                    t = pkt.get_protocol(icmp.icmp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol, icmpv4_code=t.code,
                                            icmpv4_type=t.type)
                elif protocol == in_proto.IPPROTO_TCP:
                    t = pkt.get_protocol(tcp.tcp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol, tcp_src=t.src_port, tcp_dst=t.dst_port)
                elif protocol == in_proto.IPPROTO_UDP:
                    t = pkt.get_protocol(udp.udp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol, udp_src=t.src_port, udp_dst=t.dst_port)

                # Chế độ Mitigation: Chặn nếu IP nguồn chưa bao giờ gửi ARP trên cổng này
                if self.mitigation:
                    if srcip not in self.arp_ip_to_port[dpid][in_port]:
                        self.logger.info("Cảnh báo: Phát hiện giả mạo IP %s trên cổng %s. Đang chặn...", srcip, in_port)
                        self.block_port(datapath, in_port)
                        return

                if match:
                    self.add_flow(datapath, 1, match, actions, get_flow_number(), 
                                  buffer_id=msg.buffer_id, idle=20, hard=100)
                    if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                        return

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)