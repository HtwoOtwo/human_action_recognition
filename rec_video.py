import cv2
import time

# 定义捕获器对象
cap = cv2.VideoCapture(0)

# 定义视频编写器对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('circle2.avi', fourcc, 20.0, (640, 480))

# 定义开始时间
start_time = time.time()

# 开始捕获
while cap.isOpened():
    # 捕获帧
    ret, frame = cap.read()

    if ret:
        # 将帧写入输出视频
        out.write(frame)

        # 显示帧
        cv2.imshow('frame', frame)

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 检查时间是否超过10秒
        if time.time() - start_time > 10:
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
