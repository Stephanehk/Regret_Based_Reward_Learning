export default class InputHandler {
  constructor(vehicle) {
    document.addEventListener("keydown", (event) => {
      switch (event.keyCode) {
        case 37:
          vehicle.moveLeft();
          break;
        case 38:
          vehicle.moveUp();
          break;
        case 39:
          vehicle.moveRight();
          break;
        case 40:
          vehicle.moveDown();
          break;
      }
    });

    document.addEventListener("keyup", (event) => {
      switch (event.keyCode) {
        case 37:
          if (vehicle.speed.x < 0) {
            vehicle.stop();
          }

          break;
        case 38:
          if (vehicle.speed.y < 0) {
            vehicle.stop();
          }

          break;
        case 39:
          if (vehicle.speed.x > 0) {
            vehicle.stop();
          }
          break;
        case 40:
          if (vehicle.speed.y > 0) {
            vehicle.stop();
          }
      }
    });
  }
}
