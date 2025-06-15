import cv2 as cv
import numpy as np
import os


def detect_tray(image):
    img_gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_gray_scale = cv.GaussianBlur(img_gray_scale, (7, 7), 0)

    edges = cv.Canny(img_gray_scale, 30, 100)

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=100, maxLineGap=1000)

    mask = np.zeros_like(img_gray_scale)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(mask, (x1, y1), (x2, y2), 255, 2)

    cv.imshow("Tray approx outline", mask)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=1)
    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    cv.imshow("Tray outline closed", mask_closed)
    contours, _ = cv.findContours(mask_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv.contourArea)
    tray_mask = np.zeros_like(img_gray_scale)
    cv.drawContours(tray_mask, [largest_contour], 0, 255, -1)
    cv.imshow("Tray mask", tray_mask)

    return tray_mask


def detect_coins(image):
    img_gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_gray_scale = cv.medianBlur(img_gray_scale, 3)

    cv.imshow("Gray", img_gray_scale)

    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    img_gray_scale = clahe.apply(img_gray_scale)
    cv.imshow("Gray contrast lift", img_gray_scale)

    circles = cv.HoughCircles(
        img_gray_scale, cv.HOUGH_GRADIENT, 1, 50,
        param1=270, param2=35,
        minRadius=20, maxRadius=70
    )

    return circles


def classify_coins(image, circles, tray_mask):
    coins_on_tray = {"5zl": 0, "5gr": 0}
    coins_off_tray = {"5zl": 0, "5gr": 0}

    result_img = image.copy()

    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        x, y, r = circle

        on_tray = tray_mask[y, x]

        coin_type = "5zl" if r > 30 else "5gr"

        if on_tray:
            coins_on_tray[coin_type] += 1
            color = (0, 255, 0)
        else:
            coins_off_tray[coin_type] += 1
            color = (0, 0, 255)

        cv.circle(result_img, (x, y), r, color, 2)
        cv.circle(result_img, (x, y), 2, color, 3)
        cv.putText(result_img, coin_type, (x - 20, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv.putText(result_img, f"On tray: 5zl: {coins_on_tray['5zl']}, 5gr: {coins_on_tray['5gr']}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(result_img, f"Off tray: 5zl: {coins_off_tray['5zl']}, 5gr: {coins_off_tray['5gr']}", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return result_img, coins_on_tray, coins_off_tray


def main():
    image_files = [f"tray{i}.jpg" for i in range(1, 9)]
    print(image_files)
    for image_file in sorted(image_files):

        image = cv.imread(image_file)

        tray_mask = detect_tray(image)

        circles = detect_coins(image)

        result_img, coins_on_tray, coins_off_tray = classify_coins(image, circles, tray_mask)

        cv.imshow("Result", result_img)

        print(f"Image: {image_file}")
        print(f"On tray: 5zl: {coins_on_tray['5zl']},\n5gr: {coins_on_tray['5gr']}")
        print(f"Off tray: 5zl: {coins_off_tray['5zl']},\n5gr: {coins_off_tray['5gr']}")
        print("-" * 50)

        key = cv.waitKey(0)
        if key == 27:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
