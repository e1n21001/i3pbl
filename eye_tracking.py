import cv2
import pyautogui
import numpy as np

# Haar Cascadeのロード
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# 画面のサイズを取得
screen_width, screen_height = pyautogui.size()

def main():
    # カメラの初期化
    cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラを使用することを意味します

    # 初期化
    prev_mouse_x, prev_mouse_y = 0, 0
    movement_threshold = 50  # カーソルを動かす閾値（ピクセル単位）

    while cap.isOpened():
        # フレームを1枚ずつ読み込む
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 目の検出
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(eyes) >= 2:
            # 距離の閾値を設定して、近い目のペアを見つける
            threshold_distance = 100
            eye_pairs = []

            # 目のペアを見つける
            for i in range(len(eyes)):
                for j in range(i + 1, len(eyes)):
                    (x1, y1, w1, h1) = eyes[i]
                    (x2, y2, w2, h2) = eyes[j]
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if distance < threshold_distance:
                        eye_pairs.append((eyes[i], eyes[j]))

            for (eye1, eye2) in eye_pairs:
                # 左上の点と右下の点を決定するために、ペアの目の位置を使います
                x_min = min(eye1[0], eye2[0])
                y_min = min(eye1[1], eye2[1])
                x_max = max(eye1[0] + eye1[2], eye2[0] + eye2[2])
                y_max = max(eye1[1] + eye1[3], eye2[1] + eye2[3])

                # 目のペアを囲む四角形を描画
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # 中心座標を計算
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # 画面上の位置に変換
                mouse_x = int(center_x * screen_width / frame.shape[1])
                mouse_y = int(center_y * screen_height / frame.shape[0])

                # 中心座標を出力
                print(f"Center coordinates: ({center_x}, {center_y})")
                #cv2.putText(frame, str(center_y), (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if center_y<150:
                    cv2.putText(frame, "high position", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                elif center_y>350:
                    cv2.putText(frame, "low position", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)



                # 前回の位置から大きく変わった場合のみマウスを移動
                if abs(mouse_x - prev_mouse_x) > movement_threshold or abs(mouse_y - prev_mouse_y) > movement_threshold:
                    pyautogui.moveTo(mouse_x, mouse_y)
                    prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                # 左クリック
                # pyautogui.click()

        # フレームを表示
        cv2.imshow('Eye Tracking', frame)

        # 'q'を押すと終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
