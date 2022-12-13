sec=1
cnt=0
PROC_NAME=timekeepingserver
Thread=`ps -ef | grep $PROC_NAME | grep -v "grep"`
cd /home/dna/Desktop/FaceLognew
python3 server.py
./timekeepingserver

