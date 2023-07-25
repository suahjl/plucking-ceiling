import qrcode
import os
from dotenv import load_dotenv
import telegram_send

load_dotenv()
tel_config = os.getenv('TEL_CONFIG')

img = qrcode.make('https://github.com/suahjl/plucking-ceiling')
type(img)  # qrcode.image.pil.PilImage
img.save('Output/PluckingPO_QRCode.png')


def telsendimg(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           images=[f],
                           captions=[cap])


telsendimg(
    conf=tel_config,
    path='Output/PluckingPO_QRCode.png',
    cap='https://github.com/suahjl/plucking-ceiling'
)
