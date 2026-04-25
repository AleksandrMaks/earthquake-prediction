import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer

# 1. Загрузите изображение
img = cv2.imread('map_scan.png')

# 2. Задайте опорные точки: (пиксель_x, пиксель_y) -> (долгота, широта)
pixel_coords = np.array([[120, 340], [580, 310], [350, 780]], dtype=np.float32)
geo_coords = np.array([[37.6173, 55.7558], [37.6500, 55.7600], [37.6300, 55.7400]], dtype=np.float32)

# 3. Аффинное преобразование
matrix = cv2.getAffineTransform(pixel_coords[:3], geo_coords[:3])

# 4. Функция конвертации пикселей в координаты
def pix_to_geo(px, py):
    pt = np.array([px, py, 1], dtype=np.float32)
    lon, lat = cv2.transform(pt.reshape(1, 1, 3), matrix)[0, 0]
    return lat, lon

# 5. Пример: клик мышью → сохранение в CSV
points = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        lat, lon = pix_to_geo(x, y)
        points.append((x, y, lat, lon))
        print(f"📍 Пиксель: ({x},{y}) → {lat:.5f}, {lon:.5f}")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Map', img)

cv2.imshow('Map', img)
cv2.setMouseCallback('Map', click_event)
cv2.waitKey(0)

# 6. Экспорт
pd.DataFrame(points, columns=['px', 'py', 'lat', 'lon']).to_csv('map_points.csv', index=False)