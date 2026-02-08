import yt_dlp

url = "https://www.youtube.com/watch?v=ZaKzw9tULeM"
save_path = r"C:\Users\SARJEEL\Desktop\For DAA"

ydl_opts = {
    "outtmpl": f"{save_path}/%(title)s.%(ext)s",
    "format": "mp4"
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Downloaded successfully!")
