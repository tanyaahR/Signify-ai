import cv2
import time
import mediapipe as mp
from statistics import mode

# Initialize variables and modules
ptime = 0  # Previous time for FPS calculation
mphands = mp.solutions.hands  # MediaPipe Hands module
hands = mphands.Hands(
    min_detection_confidence=0.8,  # Adjusted confidence levels for better performance
    min_tracking_confidence=0.8
)
mpdraw = mp.solutions.drawing_utils  # Drawing utilities
cap = cv2.VideoCapture(0)  # Capture video from webcam

# Variables for dynamic gesture recognition (e.g., for 'Z')
prev_hand_x = None
movement_counter = 0

def dist(p1, p2):
    """Calculate Euclidean distance between two points."""
    return int(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    h, w, c = frame.shape  # Get frame dimensions
    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Convert the BGR image to RGB before processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)  # Process the image to detect hands

    symbol = ''  # Initialize the symbol to be displayed
    ld_point = []  # List to hold all landmark positions

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_x = int(lm.x * w)
                lm_y = int(lm.y * h)
                ld_point.append((lm_x, lm_y))
            mpdraw.draw_landmarks(frame, hand_landmarks, mphands.HAND_CONNECTIONS)

            # Initialize variables for gesture recognition
            bool_list = [0, 0, 0, 0, 0]  # [Index, Middle, Ring, Pinky, Thumb]
            dist_list = [0, 0, 0]  # Distances between finger tips
            dist_from_thumb = [0, 0, 0, 0]

            # Calculate unit distance for scaling
            unit = dist(ld_point[0], ld_point[13])

            # Determine if fingers are up or down
            for i in range(8, 21, 4):  # Finger tips: 8,12,16,20
                j = int(i / 4) - 2  # Indexing for bool_list and dist_from_thumb
                dist_from_thumb[j] = dist(ld_point[i], ld_point[4])
                if ld_point[i][1] < ld_point[i - 2][1]:  # If tip is above PIP joint
                    bool_list[j] = 1  # Finger is up
                else:
                    bool_list[j] = 0  # Finger is down

            # Thumb detection
            if ld_point[4][0] > ld_point[3][0]:
                bool_list[4] = 1  # Thumb is up
            else:
                bool_list[4] = 0  # Thumb is down

            # Calculate distances between fingers
            for i in range(8, 17, 4):
                j = int(i / 4) - 2
                dist_list[j] = dist(ld_point[i], ld_point[i + 4])

            # Initialize variables for specific gesture detection
            a = 0
            a1 = 0

            ##############################
            # Alphabet Recognition Logic #
            ##############################

            # Your existing alphabet recognition logic, modified to include 'X' and 'Z'

            # Letter A
            if bool_list[0:4] == [0, 0, 0, 0] and bool_list[4] == 1 and sum(dist_list) < 1.2 * unit:
                symbol = 'A'
            # Letter B
            elif bool_list[0:4] == [1, 1, 1, 1] and bool_list[4] == 0 and sum(dist_list) < 180:
                symbol = 'B'
            # Letter C
            elif all(ld_point[i][0] > ld_point[i - 2][0] for i in range(8, 21, 4)) and bool_list[4] == 1 and dist_from_thumb[0] + 10 > dist_from_thumb[1]:
                symbol = 'C'
            # Letter D
            elif bool_list[1:5] == [0, 0, 0, 0] and dist_from_thumb.index(min(dist_from_thumb)) == 1 and dist_from_thumb.index(max(dist_from_thumb)) == 0:
                symbol = 'D'
            # Letter E
            elif bool_list == [0, 0, 0, 0, 0] and ld_point[4][0] < ld_point[8][0] and ld_point[4][1] > ld_point[8][1]:
                symbol = 'E'
                a += 1
            # Letter F
            elif bool_list[1:4] == [1, 1, 1] and bool_list[0] == 0 and bool_list[4] == 0 and dist_from_thumb.index(min(dist_from_thumb)) == 0:
                symbol = 'F'
            # Letter G
            elif any(ld_point[i][0] > ld_point[i - 2][0] for i in range(4, 9, 4)) and bool_list[0:4] == [0, 0, 0, 0] and dist_from_thumb[0] < unit:
                symbol = 'G'
                a1 += 1
            # Letter H
            elif any(ld_point[i][0] < ld_point[i - 2][0] for i in range(8, 17, 4)) and bool_list[4] == 0 and dist_list[0] < 100:
                if ld_point[16][0] > ld_point[14][0] and ld_point[20][0] > ld_point[18][0]:
                    if all(ld_point[j][1] - 10 < ld_point[8][1] for j in range(5, 8)) and dist_from_thumb[0] > 100:
                        symbol = 'H'
            # Letter I
            elif bool_list[0:4] == [0, 0, 0, 1] and bool_list[4] == 0:
                symbol = 'I'
            # Letter J (Dynamic Gesture)
            elif bool_list[0:4] == [0, 0, 0, 1] and bool_list[4] == 0:
                if prev_hand_x is not None and ld_point[20][0] - prev_hand_x > 20:
                    movement_counter += 1
                    if movement_counter > 5:
                        symbol = 'J'
                        movement_counter = 0
                else:
                    movement_counter = 0
                prev_hand_x = ld_point[20][0]
            else:
                movement_counter = 0
                prev_hand_x = None
            # Letter K
            if bool_list == [1, 1, 0, 0, 1] and 1.7 * dist_list[0] > unit and ld_point[9][0] < ld_point[4][0] < ld_point[6][0] and ld_point[4][1] < ld_point[5][1]:
                symbol = 'K'
            # Letter L
            elif bool_list[1:4] == [0, 0, 0] and dist_from_thumb[0] > 2 * unit:
                symbol = 'L'
            # Letter M
            elif bool_list == [0, 0, 0, 0, 0] and ld_point[4][0] < ld_point[8][0]:
                symbol = 'M'
            # Letter N
            elif bool_list == [0, 0, 0, 0, 0] and ld_point[4][0] < ld_point[8][0] and ld_point[4][1] < ld_point[14][1]:
                symbol = 'N'
            # Letter O
            elif bool_list == [0, 0, 0, 0, 1] and all(d < 90 for d in dist_from_thumb):
                symbol = 'O'
            # Letter P
            elif any(ld_point[i][0] > ld_point[i - 2][0] and ld_point[i + 8][0] < ld_point[i + 6][0] for i in range(8, 13, 4)):
                if ld_point[6][1] < ld_point[4][1] < ld_point[10][1]:
                    symbol = 'P'
            # Letter Q
            elif any(ld_point[i][0] > ld_point[i - 2][0] for i in range(12, 21, 4)):
                if ld_point[8][0] < ld_point[6][0] and ld_point[4][0] < ld_point[3][0] and ld_point[8][1] > ld_point[5][1] + 120:
                    symbol = 'Q'
            # Letter R
            elif bool_list[2:5] == [0, 0, 0] and dist_list[0] < 10 and bool_list[0:2] == [1, 1]:
                symbol = 'R'
            # Letter S
            elif bool_list[0:4] == [0, 0, 0, 0] and ld_point[3][0] > ld_point[4][0] > ld_point[6][0]:
                symbol = 'S'
            # Letter T
            elif bool_list == [0, 0, 0, 0, 0] and ld_point[12][0] < ld_point[4][0] < ld_point[8][0] and ld_point[4][1] < ld_point[6][1]:
                symbol = 'T'
            # Letter U
            elif bool_list == [1, 1, 0, 0, 0] and 10 < dist_list[0] < 60:
                symbol = 'U'
            # Letter V
            elif bool_list == [1, 1, 0, 0, 0] and dist_list[0] > 60 and ld_point[4][0] < ld_point[10][0]:
                symbol = 'V'
            # Letter W
            elif bool_list == [1, 1, 1, 0, 0] and dist_list[0] < 70 and dist_list[1] < 70:
                symbol = 'W'
            # Letter X
            elif bool_list == [0, 1, 0, 0, 0] and ld_point[8][1] > ld_point[6][1]:
                symbol = 'X'
            # Letter Y
            elif bool_list == [1, 0, 0, 0, 1]:
                symbol = 'Y'
            # Letter Z (Dynamic Gesture)
            elif bool_list == [0, 1, 0, 0, 0]:
                if prev_hand_x is not None and ld_point[8][0] - prev_hand_x > 20:
                    movement_counter += 1
                    if movement_counter > 5:
                        symbol = 'Z'
                        movement_counter = 0
                else:
                    movement_counter = 0
                prev_hand_x = ld_point[8][0]
            else:
                movement_counter = 0
                prev_hand_x = None

            # Adjustments based on previous conditions
            if symbol == 'J' and a1 > 0:
                symbol = 'G'
            if a > 0:
                symbol = 'E'

            ############################
            # Number Recognition Logic #
            ############################

            # Only proceed to number recognition if no alphabet symbol is detected
            if symbol == '':
                total_fingers = sum(bool_list)
                # Number 1
                if total_fingers == 1 and bool_list == [1, 0, 0, 0, 0]:
                    symbol = '1'
                # Number 2
                elif total_fingers == 2 and bool_list == [1, 1, 0, 0, 0]:
                    symbol = '2'
                # Number 3
                elif total_fingers == 3 and bool_list == [1, 1, 1, 0, 0]:
                    symbol = '3'
                # Number 4
                elif total_fingers == 4 and bool_list == [1, 1, 1, 1, 0]:
                    symbol = '4'
                # Number 5
                elif total_fingers == 5 and bool_list == [1, 1, 1, 1, 1]:
                    symbol = '5'
                # Number 6
                elif total_fingers == 1 and bool_list == [0, 0, 0, 0, 1]:
                    symbol = '6'
                # Number 7
                elif total_fingers == 2 and bool_list == [1, 0, 0, 0, 1]:
                    symbol = '7'
                # Number 8
                elif total_fingers == 3 and bool_list == [1, 1, 0, 0, 1]:
                    symbol = '8'
                # Number 9
                elif total_fingers == 4 and bool_list == [1, 1, 1, 0, 1]:
                    symbol = '9'

            # Display the recognized symbol
            cv2.putText(frame, str(symbol), (40, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    else:
        movement_counter = 0
        prev_hand_x = None

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
