# coding=utf-8
# k-means ++ for YOLOv3 anchors
# 通过k-means ++ 算法获取YOLOv3需要的anchors的尺寸
import numpy as np

# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = int(np.random.choice(boxes_num, 1))
    centroids.append(boxes[centroid_index])

    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0,n_anchors-1):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def convert2xywh(boxes):
    boxes=np.array(boxes)
    boxes_xywh=np.zeros_like(boxes)
    boxes_xywh[:,0],boxes_xywh[:,1]=(boxes[:,0]+boxes[:,2])/2.0,(boxes[:,1]+boxes[:,3])/2.0
    boxes_xywh[:,2],boxes_xywh[:,3]=(boxes[:,2]-boxes[:,0]),(boxes[:,3]-boxes[:,1])
    return boxes_xywh
def read_label(label_path):
    boxes_lines = []
    bboxes=[]
    f = open(label_path)
    for line in f:
        boxes_lines.append(line.strip())
    f.close()
    for line in boxes_lines:
        boxes=line.split(" ")[1:]

        for box_str in boxes:
            box=box_str.split(",")
            bboxes.append(list(map(int,box[0:4])))
    return convert2xywh(bboxes)
def compute_centroids(bboxes_xywh,n_anchors,loss_convergence,iterations_num,plus):
    boxes=[]
    for box in bboxes_xywh:
            boxes.append(Box(0, 0, float(box[2]), float(box[3])))
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w , centroid.h)

    # print result
    centroid_result=[]
    for centroid in centroids:
        print("k-means result：\n")
        print(centroid.w, centroid.h)
        centroid_result.append(list(map(int,[centroid.w,centroid.h])))
    return centroid_result
def scale(centroid):
    return (centroid[0]*centroid[1])**0.5


bbox=read_label(r"F:\Video_Object_detection\HeadFaceDetectionData\all_the_data_label.txt")
n_anchors = 9
loss_convergence = 1e-6
iterations_num=1000
kmeans_result=compute_centroids(bbox,n_anchors,loss_convergence,iterations_num,True)
kmeans_result.sort(key=scale)
kmeans_result=np.array(kmeans_result)
np.save('data.npy',kmeans_result)
kmeans_result_fm=np.zeros_like(kmeans_result,dtype=np.float32)
kmeans_result_fm[0:3,:],kmeans_result_fm[3:6,:],kmeans_result_fm[6:9,:]=kmeans_result[0:3,:]/8.0,kmeans_result[3:6,:]/16.0,kmeans_result[6:9,:]/32.0

kmeans_result_fm_list=[]
for i in range(n_anchors):
    kmeans_result_fm_list+=list(kmeans_result_fm[i,:])
for i in range(len(kmeans_result_fm_list)):
        kmeans_result_fm_list[i]=kmeans_result_fm_list[i]*544/1920

kmeans_result_fm_list = ['{:.2f}'.format(i) for i in kmeans_result_fm_list]
with open("pre_anchor.txt","w") as f:
    string=",".join(kmeans_result_fm_list)
    f.write(string)
