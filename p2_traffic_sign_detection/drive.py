import asyncio
import websockets
from PIL import Image
import json
import cv2
import numpy as np
import base64
from io import BytesIO

# MẶC ĐỊNH
center = 320
lane_width = 100
steering_angle = 0


def find_lane_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)
    img_canny = cv2.Canny(img_gauss, 150, 200)
    return img_canny


def birdview_transform(img):
    IMAGE_H, IMAGE_W = img.shape[:2]  # 480 640

    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H // 3], [IMAGE_W, IMAGE_H // 3]])
    # [[0, 480], [640, 480], [0, 160], [640, 160]
    # Lấy 4 góc của khung cảnh

    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    # [[240, 480], [400, 480], [-160, 0], [800, 0]
    # Xác định tọa độ 4 góc sau khi warp

    M = cv2.getPerspectiveTransform(src, dst)  # Ma trận
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))  # Image warping
    return warped_img


def find_left_right_points(image, draw=None, line=0.9):
    global center, lane_width

    IMAGE_H, IMAGE_W = image.shape[:2]

    red_line_y = int(IMAGE_H * line)

    if draw is not None:
        cv2.line(draw, (0, red_line_y), (IMAGE_W, red_line_y), (0, 0, 255), 2)

    red_line = image[red_line_y, :]

    left_point = False
    right_point = False

    # Xác định điểm trái
    for x in range(center, 0, -1):
        if red_line[x] > 0:
            left_point = x
            break
    # Xác định điểm phải
    for x in range(center + 1, IMAGE_W):
        if red_line[x] > 0:
            right_point = x
            break

    # Ước chừng điểm phải khi chưa xác định được
    if left_point and not right_point:
        right_point = left_point + lane_width

    # Ước chừng điểm trái khi chưa xác định được
    if right_point and not left_point:
        left_point = right_point - lane_width

    # Từ điểm trái và phải ta tính được trung tâm và độ dày của đường
    center = (right_point + left_point) // 2
    lane_width = right_point - left_point
    if lane_width < 50:
        lane_width = 50
    # Từ đó dựa vào những thông số này tính toán cho lần sau với kì vọng đem đến độ chính xác cao hơn

    # Vẽ 2 điểm trái phải và trung tâm
    # Trái Blue, Tâm Red, Phải Green
    if draw is not None:
        if left_point != -1:
            draw = cv2.circle(
                draw, (left_point, red_line_y), 7, (255, 0, 0), -1)
        if right_point != -1:
            draw = cv2.circle(
                draw, (right_point, red_line_y), 7, (0, 255, 0), -1)
            draw = cv2.circle(
                draw, (center, red_line_y), 7, (0, 0, 255), -1)

    return left_point, right_point


def calculate_control_signal(img, draw=None, line=0.9):
    global steering_angle
    # Tạo ảnh canny
    img_canny = find_lane_lines(img)

    # Bẻ ảnh canny sang góc nhìn từ trên xuống dưới
    img_canny_birdview = birdview_transform(img_canny)
    cv2.imshow("canny", img_canny_birdview)

    # Ảnh copy để hiển thị
    if draw is not None:
        draw[:, :] = birdview_transform(draw)

    # Dùng canny_birdview để tìm 2 điểm trái phải
    left_point, right_point = find_left_right_points(img_canny_birdview, draw=draw, line=line)

    # throttle chỉ tốc độ, ở đây max speed là 1 (50km/h)
    # steering_angle chỉ góc cua
    throttle = 2
    im_center = img.shape[1] // 2

    if left_point != -1 and right_point != -1:

        # Tính độ lệch
        center_point = (right_point + left_point) // 2
        center_diff = im_center - center_point

        # Từ độ lệch ta tính góc cua cho hợp lý
        steering_angle = - float(center_diff * 0.015)

        # Từ góc cua ta chỉnh lại tốc độ (Cua càng cao tốc độ càng thấp)
        throttle -= abs(steering_angle)
        if throttle < 0.1:
            throttle = 0.1
    else:
        steering_angle = -steering_angle

    return throttle, steering_angle


async def echo(websocket, path):
    async for message in websocket:
        # Lấy ảnh từ camera
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Tạo 1 ảnh bản sao của ảnh gốc nhận từ camera
        copy = image.copy()

        # Tính tốc độ và góc cua
        throttle_85, steering_angle_85 = calculate_control_signal(image, draw=copy,
                                                                  line=0.85)  # Tham số line để tinh chỉnh vị trí red line (0.85 là con số khá hợp lí trong mọi đường đi)

        # Hiển thị ảnh copy
        cv2.imshow("Birdview", copy)
        cv2.waitKey(1)

        # Gửi tín hiệu điều khiển về cho mô phỏng
        message = json.dumps({"throttle": throttle_85, "steering": steering_angle_85})
        print(message)
        await websocket.send(message)


async def main():
    async with websockets.serve(echo, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # chạy vô tận


asyncio.run(main())