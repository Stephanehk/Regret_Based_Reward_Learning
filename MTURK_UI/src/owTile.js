import { detectCollision } from "./collisionDetection";

export default class OWTile {
  constructor(game, position) {
    this.img = document.getElementById("img_ow_tile");
    this.blockade = document.getElementById("img_blockade");

    this.game = game;

    this.position = position;
    this.size_x = 80;
    this.size_y = 80;

    this.cellSize = game.gameWidth / 10;
    this.lastCol = { x: -1, y: -1 };
    this.isLeaving = false;
    this.leavingGoal = null;
    // this.markedForDeletion = false;
  }

  // detectIsLeaving(vehicle) {
  //   if (vehicle.find_is_leaving_ow() && this.isLeaving === false) {
  //     this.isLeaving = true;
  //     this.leavingGoal = vehicle.goal;
  //     this.game.score -= 2;
  //     this.game.gasScore -= 2;
  //     return true;
  //   }
  //   return false;
  //   // if (
  //   //   vehicle.find_is_leaving_ow() &&
  //   //   this.isLeaving === true &&
  //   //   this.leavingGoal !== vehicle.goal
  //   // ) {
  //   //   this.isLeaving = true;
  //   //   return true;
  //   // }

  //   // if (!vehicle.find_is_leaving_ow()) {
  //   //   this.isLeaving = false;
  //   // }
  //   return false;
  // }

  update() {
    if (detectCollision(this.game.vehicle, this)) {
      // if (!this.isOW(this.game.vehicle.lastCol)){
      //   this.game.gasScore-=1;
      // }
      // this.game.ball.speed.x = 0;
      // this.game.ball.speed.y = 0;
      // this.markedForDeletion = true;
      // console.log("here");
      this.game.is_in_blocking = true;
      // this.game.score -= 2;
      // this.game.gasScore -= 2;
      this.game.vehicle.updatedScore = true;
    }
  }

  isOW(cords) {
    for (var i = 0; i < this.game.ow_cords.length; i++) {
      // use array[i] here
      let ow_pos = this.game.ow_cords[i];
      if (cords.x === ow_pos.x && cords.y === ow_pos.y) {
        return true;
      }
    }
    return false;
  }

  draw(ctx) {
    ctx.drawImage(
      this.img,
      this.cellSize * this.position.x,
      this.cellSize * this.position.y,
      this.size_x,
      this.size_y
    );
    //check if above tile is a oneway tile
    // if (!this.isOW({ x: this.position.x, y: this.position.y - 1 })) {
    //   ctx.drawImage(
    //     this.blockade,
    //     this.cellSize * this.position.x - 7,
    //     this.cellSize * this.position.y - 20,
    //     75,
    //     30
    //   );
    //   // ctx.drawImage(
    //   //   this.blockade,
    //   //   this.cellSize * this.position.x + 28,
    //   //   this.cellSize * this.position.y - 20,
    //   //   40,
    //   //   30
    //   // );
    // }
    //
    // if (!this.isOW({ x: this.position.x, y: this.position.y + 1 })) {
    //   ctx.drawImage(
    //     this.blockade,
    //     this.cellSize * this.position.x - 7,
    //     this.cellSize * this.position.y + this.cellSize,
    //     75,
    //     30
    //   );
      // ctx.drawImage(
      //   this.blockade,
      //   this.cellSize * this.position.x + 30,
      //   this.cellSize * this.position.y + this.cellSize - 25,
      //   40,
      //   30
      // );
    //}
  }
}
