import cv2 as cv
import numpy as np


def detect_coins_and_tray(image_path):
    img_color = cv.imread(image_path)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (9, 9), 2)

    # Wykrywanie monet
    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, dp=1.2, minDist=50,
                              param1=100, param2=30, minRadius=10, maxRadius=100)

    coins = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for circle in circles:
            coins.append({'center': (circle[0], circle[1]), 'radius': circle[2]})

    # Wykrywanie tacy (linii prostych)
    edges = cv.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=10)

    tray_box = None
    if lines is not None:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
        points = np.array(points)
        tray_box = cv.boundingRect(points)

    # Klasyfikacja monet
    coins_on_tray = []
    coins_off_tray = []
    five_zl = []
    five_gr = []

    if tray_box:
        x, y, w, h = tray_box
        for coin in coins:
            cx, cy = coin['center']
            if x <= cx <= x + w and y <= cy <= y + h:
                coins_on_tray.append(coin)
            else:
                coins_off_tray.append(coin)
    else:
        coins_off_tray = coins  # Brak tacy wykrytej

    for coin in coins:
        if coin['radius'] >= 35:  # granica promienia dla 5 zł
            coin['type'] = '5zl'
            five_zl.append(coin)
        else:
            coin['type'] = '5gr'
            five_gr.append(coin)

    return {
        'image': image_path,
        'total_coins': len(coins),
        'on_tray': len(coins_on_tray),
        'off_tray': len(coins_off_tray),
        '5zl': len([c for c in coins if c['type'] == '5zl']),
        '5gr': len([c for c in coins if c['type'] == '5gr']),
    }


# Analiza wszystkich zdjęć
image_files = [f"tray{i}.jpg" for i in range(1, 9)]
for img in image_files:
    result = detect_coins_and_tray(img)
    print(f"--- {result['image']} ---")
    print(f"  Wszystkie monety: {result['total_coins']}")
    print(f"  Monety na tacy:   {result['on_tray']}")
    print(f"  Monety poza tacą: {result['off_tray']}")
    print(f"  5 zł:             {result['5zl']}")
    print(f"  5 gr:             {result['5gr']}")
    print()