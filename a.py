import time
import datetime
import webbrowser

# 1時間毎に任意のノートブックを開く
url = "https://colab.research.google.com/drive/12Eyl29ZYwhxSmfUHHds2yfvgieTH9_lW?hl=ja#scrollTo=rItjoukYDQHg"
for i in range(12):
    browse = webbrowser.get("chrome")
    browse.open(url)
    print(i, datetime.datetime.today())
    time.sleep(60 * 60)
