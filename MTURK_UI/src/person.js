import { detectCollision } from "./collisionDetection";

export default class Person {
  constructor(game, position) {
    this.img = document.getElementById("img_person");

    this.game = game;

    this.position = position;
    this.size_x = 50;
    this.size_y = 50;
    this.cellSize = game.gameWidth / 10;
    this.lastCol = { x: -1, y: -1 };
    // this.markedForDeletion = false;
  }

  update() {
    if (detectCollision(this.game.vehicle, this)) {
      // this.game.ball.speed.x = 0;
      // this.game.ball.speed.y = 0;
      // this.markedForDeletion = true;
      this.game.find_is_leaving_ow(this.position);
      if (!this.game.isLeaving) {
        this.game.reached_terminal = true;
        // this.game.score -= 50;
        // this.game.pScore -= 50;
        this.game.vehicle.updatedScore = true;
        this.game.nPeople +=1;
      }
    }
  }

  draw(ctx) {
    ctx.drawImage(
      this.img,
      this.cellSize * this.position.x + this.cellSize / 2 - this.size_x / 2,
      this.cellSize * this.position.y + this.cellSize / 2 - this.size_y / 2,
      this.size_x,
      this.size_y
    );
  }
}
