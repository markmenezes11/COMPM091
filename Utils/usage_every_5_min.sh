while true
do
    ps -p $@ -o %cpu,%mem,cmd
    sleep 300
done
