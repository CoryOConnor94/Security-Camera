from google.cloud import storage

client = storage.Client()
bucket = client.bucket('your_bucket_name')
blob = bucket.blob('videos/video.mp4')

video_path = 'path_to_your_video.mp4'
blob.upload_from_filename(video_path)
print("Video uploaded to Google Cloud Storage.")