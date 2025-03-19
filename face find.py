import cv2
import pygame
import tkinter.messagebox as msgbox
import os

pn = 0

msgbox.showinfo('如何退出人脸寻找窗口', '按q键退出')
msgbox.showinfo('如何查看所有人脸', '按 ← 或者 → 查看上/下一张人脸照片')


def create_directory_if_not_exists(directory_path):
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        # 如果不存在，则创建目录
        os.makedirs(directory_path)


def delete_all_files_in_directory(directory_path):
    # 确保目录存在
    if not os.path.exists(directory_path):
        return

    # 列出目录中的所有文件
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # 遍历文件列表并删除每个文件
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")


dp = 'face\\face_find'
delete_all_files_in_directory(dp)


def face_find():
    global pn
    # 加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 创建VideoCapture对象来从摄像头捕获视频
    cap = cv2.VideoCapture(0)

    # 准备保存人脸的目录
    face_save_dir = 'face\\face_find'
    if not os.path.exists(face_save_dir):
        os.makedirs(face_save_dir)

    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 将捕获的帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用CascadeClassifier检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 在检测到的人脸周围画矩形框
        for i, (x, y, w, h) in enumerate(faces):
            pn += 1
            # 绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 截取人脸区域并保存
            faceROI = gray[y:y + h, x:x + w]
            face_save_path = os.path.join(face_save_dir, f"face{i}.png")
            cv2.imwrite(face_save_path, faceROI)

        # 显示结果帧
        cv2.imshow('Frame', frame)

        # 按'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV创建的窗口
    cv2.destroyAllWindows()
    return face_save_dir


directory_path = 'face\\face_find'
create_directory_if_not_exists(directory_path)
face_find()
pn = len(os.listdir('face\\face_find'))
msgbox.showinfo('人脸', f'刚刚的画面中有{pn}张脸')

# 初始化Pygame
pygame.init()

# 设置窗口大小
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('视频中的脸')

# 设置文件夹路径
folder_path = 'face\\face_find'  # 替换为你的文件夹路径

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# 加载所有图片
images = [pygame.image.load(os.path.join(folder_path, f)).convert_alpha() for f in image_files]

# 当前显示的图片索引
current_image_index = 0

# 游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                # 切换到上一张图片
                current_image_index = (current_image_index - 1) % len(images)
            elif event.key == pygame.K_RIGHT:
                # 切换到下一张图片
                current_image_index = (current_image_index + 1) % len(images)

    # 清屏
    screen.fill((0, 0, 0))

    # 绘制当前图片
    current_image = images[current_image_index]
    current_image_rect = current_image.get_rect()
    current_image_rect.center = screen.get_rect().center
    screen.blit(current_image, current_image_rect)

    # 更新屏幕显示
    pygame.display.update()
# 退出Pygame
pygame.quit()
