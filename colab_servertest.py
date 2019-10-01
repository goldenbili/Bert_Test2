from socket import *
import sys
import threading
import time
from time import localtime

import imp

HOST = '127.0.0.1'
PORT = 1234  # 设置侦听端口
BUFSIZ = 1024


if sys.version[0] == '2':
    imp.reload(sys)
    sys.setdefaultencoding("utf-8")

class TcpServer():
    def __init__(self):
        self.ADDR = (HOST, PORT)
        try:
            self.STOP_CHAT = False
            self.sock = socket(AF_INET, SOCK_STREAM)
            print('%d is open' % PORT)

            self.sock.bind(self.ADDR)
            self.sock.listen(5)
            # 设置退出条件


            # 所有监听的客户端
            self.clients = {}
            self.thrs = {}
            self.stops = []

        except Exception as e:
            print("%d is down" % PORT)
            return None

    def listen_client(self):
      
        while not self.STOP_CHAT:
            print(u'等待接入，侦听端口:%d' % (PORT))
            self.tcpClientSock, self.addr = self.sock.accept()
            print(u'接受连接，客户端地址：', self.addr)
            address = self.addr
            # 将建立的client socket链接放到列表self.clients中
            self.clients[address] = self.tcpClientSock
            # 分别将每个建立的链接放入进程中，接收且分发消息
            self.thrs[address] = threading.Thread(target=self.readmsg, args=[address])
            self.thrs[address].start()
            time.sleep(0.5)
            #self.tcpClientSock.send(b'you are connect...')
        print(u'系統結束')



    def readmsg(self, address):
        # 如果地址不存在，则返回False
        if address not in self.clients:
            return False
        # 得到发送消息的client socket
        client = self.clients[address]
        while True:
            try:
                # 获取到消息内容data
                data = client.recv(BUFSIZ)
            except:
                print(error)
                self.close_client(address)
                break
            if not data:
                break
            # python3使用bytes，所以要进行编码
            # s='%s发送给我的信息是:[%s] %s' %(addr[0],ctime(), data.decode('utf8'))
            # 对日期进行一下格式化
            ISOTIMEFORMAT = '%Y-%m-%d %X'
            stime = time.strftime(ISOTIMEFORMAT, localtime())
            print([address], '@',[stime],':', data.decode('utf8'))

            self.STOP_CHAT = (data.decode('utf8').upper() == "QUIT")

            if self.STOP_CHAT:
                print("quit")
                self.close_client(address)
                print("already quit")
                break
             



    def close_client(self, address):
        try:
            '''
            print(u'try leave')
            client = self.clients.pop(address)
            print(u'try leave1')
            self.stops.append(address)
            print(u'try leave2')
            client.close()
            print(u'try leave3')
            '''
            for k in self.clients:
                print(u'try leave')
                print(u'try client1:', [self.clients[k]])
                print(u'try client2:', [self.clients[address]])
                print(u'try client3:', [k])
                print(u'try client4:', [address])
                client = self.clients.pop(k)
                #print(u'try leave1')
                #self.stops.append(k)
                print(u'try leave2')
                client.close()
                print(u'try leave3')
                '''
                print(u'try leave4:client:',[self.clients[k]])
                self.clients[k].send(str(address) + u"已经离开了")
                '''
        except:
            print(u'try fault')
            pass
        print(str(address) + u'已经退出')
        
        
        
tserver = None
while tserver == None:
  tserver = TcpServer()
tserver.listen_client()

print(u'系統結束')
