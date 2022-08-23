import asyncio
import websockets
import numpy as np
import cv2
import mss
import mss.tools

connected = set()
seen_worker_ids = []
data = [0,0]
async def server(websocket, path):
    lastName = None
    while True:
        async for message in websocket:
            if message not in seen_worker_ids or message == "":
                seen_worker_ids.append(message)

                sample_number = data[0]
                observationType = data[1]

                print ("sending " + str(sample_number) + "," + str(observationType))

                await websocket.send(str(sample_number) + "," + str(observationType))
                sample_number = int(sample_number)
                observationType = int(observationType)

                if observationType < 3:
                    observationType +=1
                else:
                    observationType = 0
                    sample_number+=1
                if sample_number >= 30:
                    sample_number = 0
                data[0] = str(sample_number)
                data[1] = str(observationType)

start_server = websockets.serve(server, "localhost", 3000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
