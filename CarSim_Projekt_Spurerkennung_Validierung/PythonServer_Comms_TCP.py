import socket
import threading

class TcpComms():
    def __init__(self,tcpIP,portTX,portRX,enableRX=False,suppressWarnings=True):
        """
        Constructor
        :param tcpIP: Must be string e.g. "127.0.0.1"
        :param portTX: integer number e.g. 8000. Port to transmit from i.e From Python to other application
        :param portRX: integer number e.g. 8001. Port to receive on i.e. From other application to Python
        :param enableRX: When False you may only send from Python and not receive. If set to True a thread is created to enable receiving of data
        :param suppressWarnings: Stop printing warnings if not connected to other application
        """        

        self.tcpIP = tcpIP
        self.tcpSendPort = portTX
        self.tcpRcvPort = portRX
        self.enableRX = enableRX
        self.suppressWarnings = suppressWarnings # when true warnings are suppressed
        self.isDataReceived = False
        self.dataRX = None
        self.conn = None

        # Connect via TCP
        self.tcpSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # internet protocol, tcp (stream) socket        
        self.tcpSock.bind((tcpIP, portRX))
        self.tcpSock.listen() # 1?
        print("Waiting for connection on " + tcpIP + ":" + str(portRX))
        self.conn, adr = self.tcpSock.accept()

        print("Connected by " + str(adr))

        # Create Receiving thread if required
        if enableRX:            
            self.rxThread = threading.Thread(target=self.ReadTcpThreadFunc, daemon=True)
            self.rxThread.start()

    def __del__(self):
        self.CloseSocket()

    def CloseSocket(self):
        # Function to close socket
        self.tcpSock.close()

    def SendData(self, strToSend):
        # Use this function to send string to C#
        self.conn.sendall(bytes(strToSend,'utf-8'))

    def ReceiveData(self):
        """
        Function BLOCKS until data is returned from C#. It then attempts to convert it to string and returns on successful conversion.
        An warning/error is raised if:
            - Warning: Not connected to C# application yet. Warning can be suppressed by setting suppressWarning=True in constructor
            - Error: If data receiving procedure or conversion to string goes wrong
            - Error: If user attempts to use this without enabling RX
        :return: returns None on failure or the received string on success
        """
        if not self.enableRX: # if RX is not enabled, raise error
            raise ValueError("Attempting to receive data without enabling this setting. Ensure this is enabled from the constructor")

        data = None
        try:
            # setup data and its buffersize (images will contain ca. 20.000 bytes)                        
            data = self.conn.recv(32000)            

        except WindowsError as e:
            if e.winerror == 10054: # An error occurs if you try to receive before connecting to other application
                if not self.suppressWarnings:
                    print("Are You connected to the other application? Connect to it!")
                else:
                    pass
            else:
                raise ValueError("Unexpected Error. Are you sure that the received data can be converted to a string")

        return data

    def ReadTcpThreadFunc(self): # Should be called from thread
        """
        This function should be called from a thread [Done automatically via constructor]
                (import threading -> e.g. udpReceiveThread = threading.Thread(target=self.ReadUdpNonBlocking, daemon=True))
        This function keeps looping through the BLOCKING ReceiveData function and sets self.dataRX when data is received and sets received flag
        This function runs in the background and updates class variables to read data later

        """

        self.isDataReceived = False # Initially nothing received
        
        while True:
            data = self.ReceiveData()  # Blocks (in thread) until data is returned (OR MAYBE UNTIL SOME TIMEOUT AS WELL)
            self.dataRX = data # Populate AFTER new data is received
            self.isDataReceived = True
            # When it reaches here, data received is available

    def ReadReceivedData(self):
        """
        This is the function that should be used to read received data
        Checks if data has been received SINCE LAST CALL, if so it returns the received string and sets flag to False (to avoid re-reading received data)
        data is None if nothing has been received
        :return:
        """

        data = None

        if self.isDataReceived: # if data has been received
            self.isDataReceived = False
            data = self.dataRX
            self.dataRX = None # Empty receive buffer

        return data