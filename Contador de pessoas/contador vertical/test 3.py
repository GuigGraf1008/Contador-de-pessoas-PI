import cv2

def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('2.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

detects = []

# Obter altura do vídeo
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Centralizar a linha de detecção verticalmente
posH = height // 2

offset = 50  # Deslocamento permitido

xy1 = (300, posH)  # Ponto inicial da linha
xy2 = (800, posH)  # Ponto final da linha

total = 0
up = 0
down = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    dilation = cv2.dilate(opening, kernel, iterations=8)

    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=8)

    cv2.line(frame, xy1, xy2, (255, 0, 0), 3)

    cv2.line(frame, (xy1[0], posH - offset), (xy2[0], posH - offset), (255, 255, 0), 2)

    cv2.line(frame, (xy1[0], posH + offset), (xy2[0], posH + offset), (255, 255, 0), 2)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)

        if int(area) > 3000:
            centro = center(x, y, w, h)

            cv2.putText(frame, str(i), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(detects) <= i:
                detects.append([])
            if centro[1] > posH - offset and centro[1] < posH + offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            i += 1

    if i == 0:
        detects.clear()

    i = 0

    if len(contours) == 0:
        detects.clear()
    else:
        for detect in detects:
            for (c, l) in enumerate(detect):
                if detect[c - 1][1] < posH and l[1] > posH:
                    detect.clear()
                    up += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 255, 0), 5)
                    continue

                if detect[c - 1][1] > posH and l[1] < posH:
                    detect.clear()
                    down += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                    continue

                if c > 0:
                    cv2.line(frame, detect[c - 1], l, (0, 0, 255), 1)

    cv2.putText(frame, "TOTAL: " + str(total), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, "UP: " + str(up), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "DOWN: " + str(down), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()