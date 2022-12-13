import cv2
import subprocess
import numpy as np
import os
import time


class FFmpegVideoCapture:
    def __init__(self, source, width, height, mode="bgr24", start_seconds=0, duration=0, verbose=False):
        x = ['ffmpeg']
        if start_seconds > 0:
            x.append("-accurate_seek")
            x.append("-ss")
            x.append("%f" % start_seconds)
        if duration > 0:
            x.append("-t")
            x.append("%f" % duration)
        x.extend(['-i', source,
                  "-f", "rawvideo",
                  "-pix_fmt", mode,
                  # "-vf", "vidstabdetect=stepsize=6:shakiness=10:accuracy=15",
                  "-"])
        print(x)
        self.nulldev = open(os.devnull, "w") if not verbose else None
        self.ffmpeg = subprocess.Popen(x, stdout=subprocess.PIPE, stderr=subprocess.STDERR if verbose else self.nulldev)
        self.width = width
        self.height = height
        self.mode = mode
        if self.mode == "gray":
            self.fs = width * height
        elif self.mode == "yuv420p":
            self.fs = width * height * 6 / 4
        elif self.mode == "rgb24" or self.mode == "bgr24":
            self.fs = width * height * 3
        self.output = self.ffmpeg.stdout

    def read(self):
        if self.ffmpeg.poll():
            return False, None
        x = self.output.read(self.fs)
        if x == "":
            return False, None
        if self.mode == "gray":
            return True, np.frombuffer(x, dtype=np.uint8).reshape((self.height, self.width))
        elif self.mode == "yuv420p":
            k = self.width * self.height
            return True, (np.frombuffer(x[0:k], dtype=np.uint8).reshape((self.height, self.width)),
                          np.frombuffer(x[k:k + (k / 4)], dtype=np.uint8).reshape((self.height / 2, self.width / 2)),
                          np.frombuffer(x[k + (k / 4):], dtype=np.uint8).reshape((self.height / 2, self.width / 2))
                          )
        elif self.mode == "bgr24" or self.mode == "rgb24":
            return True, (np.frombuffer(x, dtype=np.uint8).reshape((self.height, self.width, 3)))


if __name__ == '__main__':
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    s = '/dev/video0'
    capture = FFmpegVideoCapture(s, 640, 480)
    while True:
        st = time.time()
        ret, img = capture.read()
        if not ret:
            break

        cv2.imshow("img", img)
        cv2.waitKey(1)
        print(1/(time.time()-st))

