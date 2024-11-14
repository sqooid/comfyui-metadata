import sys
from server import PromptServer
from aiohttp import web

routes = PromptServer.instance.routes


@routes.get("/sq/restart")
async def restart(request):
    sys.exit(69)
