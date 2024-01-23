# -*- coding: utf-8 -*-
import cv2
import numpy as np

def img_binary_df(weigher_img):
    raw_img = cv2.imread(weigher_img, cv2.IMREAD_COLOR)
    #img = raw_img[200:280,280:560,:]# original
    img = raw_img[180:280,280:590,:]
    b_channel,g_channel,r_channel = cv2.split(img)
    if np.max(r_channel) < 200: # changed at 20220420
        return
    # if np.max(b_channel) <200 or np.max(g_channel) < 200 or np.max(r_channel) < 200:
    #     return 
    # 要用於形態學閉運算的 kernel
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 將影像轉為灰階
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, MORPH_KERNEL)
    cv2.bitwise_not(img_binary, img_binary)
    return img_binary

def img_binary_for_high_light(weigher_img):
    raw_img = cv2.imread(weigher_img, cv2.IMREAD_COLOR)
    #img = raw_img[200:280,280:560,:]# original
    img = raw_img[180:280,280:590,:]
    # 要用於形態學閉運算的 kernel
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 將影像轉為灰階
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    img_binary = cv2.erode(img_binary,MORPH_KERNEL)
    cv2.bitwise_not(img_binary, img_binary)
    return img_binary


def detect_number(mnist_model,img_binary):
    IMG_BORDER = 5  # 要忽略辨識的影像邊緣寬度
    DETECT_THRESHOLD = 0.7  # 顯示預測結果的門檻 (70%)
    LABEL_SIZE = 0.7  # 顯示預測值的字體比例 (70%)
    #RUNTIME_ONLY = True  # 使用 TF Lite Runtime
    RUNTIME_ONLY = False  # 使用 TF Lite Runtime
    
    # 載入 TF Lite 模型
    if RUNTIME_ONLY:
        from tflite_runtime.interpreter import Interpreter
        interpreter = Interpreter(model_path=mnist_model)
    else:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=mnist_model)
    
    # 準備模型
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 讀取輸入長寬
    INPUT_H, INPUT_W = input_details[0]['shape'][1:3]
    IMG_H, IMG_W = img_binary.shape[:2]

    # 圈出畫面中的輪廓
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort the contures by the x of each coutours
    contours_sorted = sorted(contours, key = lambda i:cv2.boundingRect(i)[0])

    '''
    only for test
    '''
    # 拷貝影像 (以便展示結果)
    img_binary_copy = img_binary.copy()
    img_binary_result_copy = img_binary.copy()
    
    # retrun weigher_value
    weigher_value = ''
    # 走訪所有輪廓

    for contour in contours_sorted:
        x, y, w, h = cv2.boundingRect(contour)

        # 在輪廓上畫框
        #cv2.rectangle(img_binary_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        #cv2.rectangle(img, (IMG_BORDER,IMG_BORDER ), ((IMG_W-1)-IMG_BORDER,(IMG_H-1)-IMG_BORDER ), (0, 0, 255), 2)
        #cv2.rectangle(img, (15, 15), (20+8, 20+35), (0, 255, 255), 2)
        #cv2.rectangle(img, (15, 15), (20+50, 20+70), (0, 255, 255), 2)
        
        # 如果輪廓太靠近邊緣，忽略它
        if x < IMG_BORDER or x + w > (IMG_W - 1) - IMG_BORDER or y < IMG_BORDER or y + h > (IMG_H - 1) - IMG_BORDER:
            continue
    
        # 如果輪廓太大或太小，也忽略它
        if w < 8 or h < 35 or w > 50 or h > 70:
            continue
    
        # 擷取出輪廓內的影像
        img_digit = img_binary[y: y + h, x: x + w]
        
        # 在該影像周圍補些黑邊
        r = max(w, h)
        y_pad = ((w - h) // 2 if w > h else 0) + r // 5
        x_pad = ((h - w) // 2 if h > w else 0) + r // 5
        img_digit = cv2.copyMakeBorder(img_digit, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # 調整影像為模型輸入大小
        img_digit = cv2.resize(img_digit, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    
        # 做預測
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img_digit, axis=0))
        interpreter.invoke()
        predicted = interpreter.get_tensor(output_details[0]['index']).flatten()
        #print(predicted)
        # 讀取預測標籤及其概率
        label = predicted.argmax(axis=0)
        prob = predicted[label]
        
        #若概率低於門檻就忽略之
        #if prob < DETECT_THRESHOLD:
            #continue

        # 印出預測結果、概率與數字的位置
        #print(f'Detected digit: [{label}] at x={x}, y={y}, w={w}, h={h} ({prob*100:.3f}%)')
        weigher_value += str(label)
        
        '''
        only for test
        '''
        # 在另一個影像拷貝的數字周圍畫框以及顯示標籤
        cv2.rectangle(img_binary_result_copy, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.putText(img_binary_result_copy, str(label), (x + w // 10, y - h // 10), cv2.FONT_HERSHEY_COMPLEX, LABEL_SIZE, (255, 255, 255), 1)
    '''
    only for test
    '''
    cv2.imshow('',img_binary_result_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('result.jpg',img_binary_result_copy)
    if weigher_value == '':
        return 
    return weigher_value

def detect_weigher(mnist_model,weigher_img):
    img_binary = img_binary_df(weigher_img)
    if type(img_binary) != np.ndarray:
        return 
    weigher_value = detect_number(mnist_model,img_binary)
    if weigher_value == None:
        print('Using img_binary_for_high_light() function again')
        img_binary = img_binary_for_high_light(weigher_img)
        weigher_value = detect_number(mnist_model,img_binary)
    return weigher_value


if __name__ == '__main__':
    TEST_FILE = './weigher/103.124.74.8_7105_2022-04-19-10-31-09.jpg'
    #have number but detect None in first method
    #TEST_FILE = './weigher/103.124.74.8_7105_2022-03-30-15-26-38.jpg'
    #dark and no number
    #TEST_FILE = './weigher/2022-03-22/7105/103.124.74.8_7105_2022-03-22-07-56-34.jpg'
    
    TF_LITE_MODEL = './mnist.tflite'  # TF Lite 模型
    weigher_value = detect_weigher(TF_LITE_MODEL,TEST_FILE)
    print(weigher_value)

    
    
