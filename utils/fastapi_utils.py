import asyncio


async def astreamer(generator):
    try:
        for i in generator:
            yield (i)
            await asyncio.sleep(0.1)
    except asyncio.CancelledError as e:
        print("cancelled")
