import qrcode

img = qrcode.make('https://github.com/suahjl/plucking-ceiling')
type(img)  # qrcode.image.pil.PilImage
img.save('Output/PluckingPO_QRCode.png')
