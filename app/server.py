import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from tfmr_extensions import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

# docker build -t fastai-v3 . && docker run --rm -it -p 5000:5000 -v /Users/adamschiller/Projects/DeepLearning/fastai-v3/:/var/app fastai-v3

export_file_url = 'https://www.dropbox.com/s/j4hwkwoef0szzvk/export.pkl?dl=1'
export_file_name = 'export.pkl'

spm_file_url = 'https://www.dropbox.com/s/scaz74vg31zmqlo/spm_full_10k.model?dl=1'
spm_file_name = 'spm_full_10k.model'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(spm_file_url, path / 'models' / spm_file_name)
    await download_file(export_file_url, path / 'models' / export_file_name)
    try:
        learn = load_learner(path / 'models', export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment. \
                      \n\nPlease update the fastai library in your training environment and export your model again. \
                      \n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()

    try:
        img = open_image(path / 'static' / 'images' / img_data['filename'])
    except KeyError:
        img_bytes = await (img_data['file'].read())
        img = open_image(BytesIO(img_bytes))

    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
