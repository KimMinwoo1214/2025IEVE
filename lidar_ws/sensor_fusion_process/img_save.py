import cv2

def capture_and_save():
    cap = cv2.VideoCapture(2)  # 기본 카메라(웹캠) 실행
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    count = 0  # 이미지 저장 번호
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow("Camera", frame)  # 영상 출력
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' 키를 누르면 사진 저장
            filename = f"captured_image_{count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            count += 1
        elif key == ord('q'):  # 'q' 키를 누르면 종료
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save()

