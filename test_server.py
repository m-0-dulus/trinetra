from aiohttp import web

async def home(request):
    return web.Response(text="TRINETRA SERVER IS WORKING!")

app = web.Application()
app.router.add_get("/", home)
app.router.add_get("/phone", home)

print("Starting test server...", flush=True)

web.run_app(
    app,
    host="0.0.0.0",
    port=8080
)
