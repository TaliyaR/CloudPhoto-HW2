import base64
import io
import json
import os
import uuid
import boto3
import requests as r
from PIL import Image
from requests.structures import CaseInsensitiveDict

images_types = ['.jpg', '.jpeg']

AWS_ACESS_KEY_ID = os.environ.get('aws_access_key_id')
AWS_SECRET_ACESS_KEY = os.environ.get('aws_secret_access_key')
API_KEY = os.environ.get('api_key')
MESSAGE_QUEUE_URL = os.environ.get('message_queue_url')


def main(event, context):
    bucket_id = event['messages'][0]['details']['bucket_id']
    object_id = event['messages'][0]['details']['object_id']

    if (("/faces/" not in object_id) and (
            object_id.endswith(images_types[0]) or object_id.endswith(images_types[1])) and ("/" in object_id)):
        session = boto3.session.Session()

        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=AWS_ACESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACESS_KEY
        )
        sqs = session.resource(
            service_name='sqs',
            endpoint_url='https://message-queue.api.cloud.yandex.net',
            aws_access_key_id=AWS_ACESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACESS_KEY,
            region_name='ru-central1'
        )

        message_queue = sqs.Queue(url=MESSAGE_QUEUE_URL)
        s3_response_object = s3.get_object(Bucket=bucket_id, Key=object_id)
        object_content = s3_response_object['Body'].read()
        encoded_object = base64.b64encode(object_content)

        request_json = json.dumps(
            {"analyze_specs": [{"content": encoded_object.decode('ascii'), "features": [{"type": "FACE_DETECTION"}]}]},
            indent=4
        )

        authorization_headers = CaseInsensitiveDict()
        authorization_headers["Authorization"] = "Api-Key " + API_KEY
        authorization_headers["Content-Type"] = "application/json"
        faces_response = r.post(
            "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze",
            headers=authorization_headers,
            data=request_json
        )
        faces_response_json = faces_response.json()

        try:
            faces_json = faces_response_json['results'][0]['results'][0]['faceDetection']['faces']
            pillow_image = Image.open(io.BytesIO(object_content))
            faces_objects_names = []
            for (index, face) in enumerate(faces_json):
                coordinate_1 = face['boundingBox']['vertices'][0]
                coordinate_2 = face['boundingBox']['vertices'][2]
                cropped_image = pillow_image.crop((int(coordinate_1['x']), int(coordinate_1['y']),
                                                   int(coordinate_2['x']), int(coordinate_2['y'])))
                imgByteArr = io.BytesIO()
                cropped_image.save(imgByteArr, format='jpeg')
                imgByteArr = imgByteArr.getvalue()
                album_name = object_id[:object_id.find('/')]
                object_name = object_id[object_id.rfind('/') + 1:]
                object_storage_name = f"{album_name}/faces/{object_name}-{uuid.uuid1()}.jpeg"
                s3.upload_fileobj(io.BytesIO(imgByteArr), bucket_id, object_storage_name)
                faces_objects_names.append(object_storage_name)

            faces_objects_names_string = str(faces_objects_names)
            message_body = '{"faces": ' + faces_objects_names_string + ', "parentObject": "' + object_id + '"}'
            message_body = message_body.replace("'", "\"")
            message_queue.send_message(MessageBody=message_body)

        except KeyError:
            return
