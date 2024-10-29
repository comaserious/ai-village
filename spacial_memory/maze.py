import numpy as np
import matplotlib.pyplot as plt

map_matrix = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 11, 11, 11, 11, 11, 1, 6, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 2, 1, 11, 11, 11, 11, 11, 1, 6, 6, 1, 0, 0, 0, 0, 3, 3, 0, 0, 1],
    [1, 0, 8, 2, 1, 11, 11, 11, 8, 11, 1, 8, 6, 1, 0, 0, 0, 0, 8, 3, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 10, 8, 10, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 7, 7, 7, 7, 1, 0, 0, 0, 0, 0, 10, 10, 10, 1],
    [1, 0, 0, 9, 9, 9, 0, 0, 1, 7, 7, 7, 7, 1, 0, 0, 0, 0, 0, 8, 10, 10, 1],
    [1, 0, 0, 9, 9, 8, 0, 0, 1, 7, 8, 7, 7, 1, 0, 0, 0, 0, 0, 10, 10, 10, 1],
    [1, 0, 0, 9, 9, 9, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 10, 10, 10, 10, 1],
    [1, 0, 0, 9, 9, 9, 0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 0, 10, 10, 8, 10, 1],
    [1, 5, 5, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 8, 4, 0, 0, 0, 0, 0, 1],
    [1, 5, 8, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# 1 – Wall
# 0 – Path
# 2 – Joy_home
# 3 – Sadness_home
# 4 – Anger_home
# 5 – Fear_home    
# 6 – Disgust_home 
# 7 – Discussion Room
# 8 – Entrance
# 9 – Cafe
# 10 – Park
# 11 – Library

# Zone Information
zone_labels = {
    1: "Wall",
    0: "Path",
    2: "Joy_home",
    3: "Sadness_home",
    4: "Anger_home",
    5: "Fear_home",
    6: "Disgust_home",
    7: "Discussion Room",
    8: "Entrance",
    9: "Cafe",
    10: "Park",
    11: "Library"
}



map_array = np.array(map_matrix)


# 색상 맵 정의
cmap = plt.cm.get_cmap('tab10', np.max(map_array) + 1)

# 이미지로 시각화
plt.imshow(map_array, cmap=cmap, interpolation='nearest')
plt.colorbar()  # 색상 바 추가
plt.title("Maze Visualization")
plt.axis('off')  # 축 숨기기

# 구역 레이블 추가


plt.show()


def get_zone_data():
    return map_array, zone_labels

