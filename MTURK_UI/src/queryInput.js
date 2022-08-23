export default class QueryInputHandler {
  constructor() {
    document.addEventListener("keydown", (event) => {
      switch (event.keyCode) {
        case 37:
          if (window.begunQueries) {
            window.qm.pressed = true;
            window.qm.queried("left");
          }
          break;
        case 38:
        if (window.begunQueries) {
          window.qm.pressed = true;
          window.qm.queried("same");
        }
          break;
        case 39:
          if (window.begunQueries) {
            window.qm.pressed = true;
            window.qm.queried("right");
          }
          break;

        case 40:
        if (window.begunQueries) {
          window.qm.pressed = true;
          window.qm.queried("dis");
        }
        break;
      }
    });
    //
    // document.addEventListener("keyup", (event) => {
    //   switch (event.keyCode) {
    //     case 37:
    //       if (vehicle.speed.x < 0) {
    //         vehicle.stop();
    //       }
    //
    //       break;
    //     case 38:
    //       if (vehicle.speed.y < 0) {
    //         vehicle.stop();
    //       }
    //
    //       break;
    //     case 39:
    //       if (vehicle.speed.x > 0) {
    //         vehicle.stop();
    //       }
    //       break;
    //     case 40:
    //       if (vehicle.speed.y > 0) {
    //         vehicle.stop();
    //       }
    //   }
    // });
  }
}
