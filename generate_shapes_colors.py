import os
import numpy as np
import cv2
import random
from sklearn.neighbors import BallTree

from matplotlib import pyplot as plt

MIN_AREA = 400 # the bounding box's area

def generate_corners(W, ratio=0.1):
    x1 = random.sample(range(W), 1)[0]
    x2 = x1 + random.sample(range(int(W*ratio)), 1)[0] + 1 # at least 1 pixel further
    
    return x1, x2

def add_circle(img, ratio=0.1):
    
    while True:
        
        row, col = random.sample(range(W), 2)
        color, color_label = random_color()
        r = random.sample(range(int(W*ratio)), 1)[0] + 1
        y1 = col - r
        y2 = col + r
        x1 = row - r
        x2 = row + r
        
        area = (x2 - x1) * (y2 - y1)
        print (area)
        if area > MIN_AREA:
            
            break
    # -1 means filled    
    cv2.circle(img, (row, col), r, color, -1)
    return img, color_label, x1, x2, y1, y2

def random_color():
    c = random.choice((0,1,2))
    arr = np.zeros(3)
    arr[c] = 255
    # cv2 takes tuple
    return tuple(arr), c

def add_rectangle(img):
    H, W,_ = img.shape
    while True:
        x1, x2 = generate_corners(W)
        y1, y2 = generate_corners(H)
        
        area = (x2 - x1) * (y2 - y1)
        if area > MIN_AREA:
            break 
    color, color_label = random_color()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img, color_label, x1-3, x2+3, y1-3, y2+3 # not to tight

def not_being_too_close(q2, dist):
    q2 = q2[dist >= np.percentile(dist, 80)] # higher than 80% of samples
    return q2

def come_up_with_co(x1, y1, r, co, tree):
    
    q2, dist = tree.query_radius([[x1, y1]], r=r, return_distance=True)
    q2 = q2[0]
    dist = dist[0]
    
    coq2 = np.array([co[item] for item in q2])
    # avoid points on the same plane
    mask = np.squeeze(np.dstack([coq2[:,0] != x1, coq2[:,1] != y1]))
    
    try:
        mask = np.all(mask, axis=1)
    except IndexError:
        redo = True
    else:
        redo = False
        q2 = q2[mask]
        dist = dist[mask]
        
        q2 = not_being_too_close(q2, dist)
        
    if redo:
        return (redo, None)
    else:
        return (redo, q2)
    
def add_triangle(img, tree, co, W):
    
    while True:
        redo = True
        while redo:
            x1, y1 = generate_corners(W)
            ans = come_up_with_co(x1, y1, 30,co, tree)
            redo = ans[0]
            
        q2 = ans[1]
        
        redo = True
        while redo:
            vertices2 = random.sample(list(q2), 1)
            x2, y2 = co[vertices2[0]]
            ans = come_up_with_co(x2, y2, 30, co, tree)
            redo = ans[0]
        q3 = ans[1]
        
        vertices3 = random.sample(set(q3), 1)
        x3, y3 = co[vertices3[0]]
        
        #make sure the datatype is correct for cv
        vertices = np.array([[x1,y1], [x2, y2], [x3, y3]], dtype=np.int32)
        pts = vertices.reshape((-1,1,2))
        color, color_label = random_color()
        
        
        bbx1 = min([x1, x2, x3])
        bby1 = min([y1,y2, y3])
        bbx2 = max([x1, x2, x3])
        bby2 = max([y1,y2, y3])
        
        area = (x2 - x1) * (y2 - y1)
        if area > MIN_AREA:
            break
        
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
    cv2.fillPoly(img, [pts], color=color)
    return img, color_label, bbx1, bbx2, bby1, bby2

H,W = 224, 224
MAX_SAMPLES = 40

output_dir = r'/Users/wilsonfok/Documents/scripts/Others/Davinci/jupyter/temp/'

shape_list = []
color_list = []

color_dict = {0:'red',
              1:'green',
              2:'blue'}

# setup a grid
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x,y)

co = np.dstack([X.flatten(), Y.flatten()])
co = np.squeeze(co)
tree = BallTree(co)


for counter in range(MAX_SAMPLES):
    img = np.zeros((H,W,3))
    
    
    img, color_label, x1, x2, y1, y2 = add_circle(img)
    img, color_label, x1, x2, y1, y2 = add_rectangle(img)
    img, color_label, x1, x2, y1, y2 = add_triangle(img, tree, co, W)
    
    plt.figure()
    plt.imshow(img)
    
#    plt.show()
    plt.savefig(os.path.join(output_dir, str(counter) + '.png'),
                transparent=True)
    
    del img
    
        
        
