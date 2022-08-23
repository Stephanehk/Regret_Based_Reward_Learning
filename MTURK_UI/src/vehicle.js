export default class Vehicle {
  //FIX IS ENTERING BUG!!!!
  constructor(game, isAnimating) {
    this.game = game;
    this.isAnimating = isAnimating;

    this.loadRewardFunction().then(() => {
      this.loadedRFunc = true;
    });

    this.img = document.getElementById("img_car_side");
    this.scale_factor = 10 / 10;
    this.max_speed = { x: 0.1, y: 0.1 };
    this.speed = { x: 0, y: 0 };
    this.position = game.spawnPoint;
    this.curStatePrevCords = game.spawnPoint;

    this.goal = this.position;
    this.cellSize = game.gameWidth / 10;

    this.gameWidth = game.gameWidth;
    this.gameHeight = game.gameHeight;
    this.i = 0;
    this.lastCol = { x: -1, y: -1 };
    this.just_spawned = true;
    this.isDone = false;
    this.reachedGoal = false;
    this.prevGoalLCol = [null, null];
    this.updatedScore = false;
    this.lastAction = null;

    this.sideSizeX = this.scale_factor * this.cellSize;
    this.sideSizeY = this.scale_factor * this.cellSize * (4 / 5);

    this.backSizeX = this.scale_factor * this.cellSize;
    this.backSizeY = this.scale_factor * this.cellSize;
    this.size_x = this.sideSizeX;
    this.size_y = this.sideSizeY;
  }

  async loadRewardFunction() {
    // let rjson = require("/assets/trajData/" +
    //   String(this.game.boardName) +
    //   "_rewards_function.json");
    // let rjson = await (
    //   await fetch(
    //     "/assets/boards/" +
    //       String(this.game.boardName) +
    //       "_rewards_function.json"
    //   )
    // ).json();
    // console.log("loading board with name: " + String(this.boardName));

    let rjson = await (
      await fetch('https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/boards/' + String(this.game.boardName) + '_rewards_function.json')
    ).json();
    //console.log(json)

    // console.log(rjson);
    this.rFunc = [];
    for (var i in rjson) {
      let row = [];
      for (var j in rjson[i]) {
        let stateRs = [];
        for (var r in rjson[i][j]) {
          stateRs.push(rjson[i][j][r]);
        }
        row.push(stateRs);
      }
      this.rFunc.push(row);
      // this.keys.push(i);
    }
  }

  inBlocking(cords) {
    for (var i = 0; i < this.game.blockingCords.length; i++) {
      if (
        cords.x === this.game.blockingCords[i].x &&
        cords.y === this.game.blockingCords[i].y
      ) {
        return true;
      }
    }
    return false;
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
  //
  // calcCellPos() {
  //   this.x2draw =
  //     this.cellSize * this.position.x +
  //     (this.position.x + this.cellSize) / 2 -
  //     this.size_x / 2;

  //   this.y2draw =
  //     this.cellSize * this.position.y +
  //     (this.position.y + this.cellSize) / 2 -
  //     this.size_y / 2;
  // }

  isdone() {
    let ib = this.find_is_blocking();
    //
    if (ib) {
      this.isDone = true;
      this.game.trajHandler.incompleteCords.push(this.lastCol);
      this.game.trajHandler.incompleteActions.push(this.lastAction);
      // console.log(this.lastAction);
      return true;
    }
    if (
      (this.position.x - this.goal.x < 0.1 &&
        this.position.y - this.goal.y < 0.1 &&
        this.position.x - this.goal.x >= -0.01 &&
        this.position.y - this.goal.y >= -0.01) ||
      (this.goal.x - this.position.x === 0 &&
        this.goal.y - this.position.y === 0)
    ) {
      this.position = this.goal;
      this.isDone = true;
      return true;
    }
    // if (this.i < 10) {
    //   console.log(this.goal);
    //   console.log(this.position);
    //   this.i += 1;
    // }
    this.isDone = false;
    return false;
  }

  moveLeft() {
    // console.log(this.isdone())
    // this.game.isLeaving = false;
    if (this.isdone() && !this.game.reached_terminal) {
      this.lastAction = "left";
      let posGoal = { x: this.position.x - 1, y: this.position.y };

      this.game.find_is_leaving_ow(posGoal);

      //log player activity during each game
      if (this.game.isLeaving){
        this.game.triedLeaving = true
      }else if (this.position.x - 1 < 0){
        this.game.hitEdge = true
      }else if (this.inBlocking(posGoal)){
        this.game.hitHouse = true
      }


      if (
        !this.game.isLeaving &&
        this.position.x - 1 >= 0 &&
        !this.inBlocking(posGoal)
      ) {
        this.reachedGoal = false;
        this.speed.x = -this.max_speed.x;
        this.goal = { x: this.position.x - 1, y: this.position.y };
      } else {
        this.goal = { x: this.position.x, y: this.position.y };
        this.game.trajHandler.incompleteCords.push(this.goal);
        this.game.trajHandler.incompleteActions.push("left");
        this.incAction = true;
        this.lastAct = { x: -1, y: 0 };
      }

      this.img = document.getElementById("img_car_side_f");
      this.size_x = this.sideSizeX;
      this.size_y = this.sideSizeY;
    }
  }

  moveRight() {
    // this.game.isLeaving = false;
    if (this.isdone() && !this.game.reached_terminal) {
      this.lastAction = "right";
      let posGoal = { x: this.position.x + 1, y: this.position.y };

      this.game.find_is_leaving_ow(posGoal);

      //log player activity during each game
      if (this.game.isLeaving){
        this.game.triedLeaving = true
      }else if (this.position.x + 1 >= 10){
        this.game.hitEdge = true
      }else if (this.inBlocking(posGoal)){
        this.game.hitHouse = true
      }

      if (
        !this.game.isLeaving &&
        this.position.x + 1 < 10 &&
        !this.inBlocking(posGoal)
      ) {
        this.reachedGoal = false;
        this.speed.x = this.max_speed.x;
        this.goal = { x: this.position.x + 1, y: this.position.y };
      } else {
        this.goal = { x: this.position.x, y: this.position.y };
        this.game.trajHandler.incompleteCords.push(this.goal);
        this.game.trajHandler.incompleteActions.push("right");
        this.incAction = true;
        this.lastAct = { x: 1, y: 0 };
      }
      this.img = document.getElementById("img_car_side");
      this.size_x = this.sideSizeX;
      this.size_y = this.sideSizeY;
    }
  }

  moveUp() {
    //if we are animating stop car from running into our label
    if (this.isAnimating && this.position.y - 1 < 3) {
      return;
    }
    if (this.isdone() && !this.game.reached_terminal) {
      this.lastAction = "up";
      let posGoal = {
        x: this.position.x,
        y: this.position.y - 1
      };
      this.game.find_is_leaving_ow(posGoal);

      //log player activity during each game
      if (this.game.isLeaving){
        this.game.triedLeaving = true
      }else if (this.position.y - 1 < 0){
        this.game.hitEdge = true
      }else if (this.inBlocking(posGoal)){
        this.game.hitHouse = true
      }

      if (
        !this.game.isLeaving &&
        this.position.y - 1 >= 0 &&
        !this.inBlocking(posGoal)
      ) {
        this.reachedGoal = false;
        this.speed.y = -this.max_speed.y;
        this.goal = posGoal;
      } else {
        this.goal = { x: this.position.x, y: this.position.y };
        this.game.trajHandler.incompleteCords.push(this.goal);
        this.game.trajHandler.incompleteActions.push("up");
        this.incAction = true;
        this.lastAct = { x: 0, y: -1 };
      }

      this.img = document.getElementById("img_car_back");
      this.size_x = this.backSizeX;
      this.size_y = this.backSizeY;
    }
  }

  moveDown() {
    // this.game.isLeaving = false;
    if (this.isdone() && !this.game.reached_terminal) {
      this.lastAction = "down";
      let posGoal = { x: this.position.x, y: this.position.y + 1 };

      this.game.find_is_leaving_ow(posGoal);

      if (this.game.isLeaving){
        this.game.triedLeaving = true
      }else if (this.position.y + 1 >= 10){
        this.game.hitEdge = true
      }else if (this.inBlocking(posGoal)){
        this.game.hitHouse = true
      }

      if (
        !this.game.isLeaving &&
        this.position.y + 1 < 10 &&
        !this.inBlocking(posGoal)
      ) {
        this.reachedGoal = false;
        this.speed.y = this.max_speed.y;
        this.goal = posGoal;
        // console.log(this.goal);
      } else {
        this.goal = { x: this.position.x, y: this.position.y };
        this.game.trajHandler.incompleteCords.push(this.goal);
        this.game.trajHandler.incompleteActions.push("down");
        this.incAction = true;
        this.lastAct = { x: 0, y: 1 };
      }
      this.img = document.getElementById("img_car_front");
      this.size_x = this.backSizeX;
      this.size_y = this.backSizeY;
    }
  }

  stop() {
    if (this.isdone()) {
      this.speed.x = 0;
      this.speed.y = 0;
    }
  }

  // clearCols() {
  //   this.game.gameObjects.forEach(
  //     (object) => (object.lastCol = { x: -1, y: -1 })
  //   );
  // }

  // find_is_leaving_ow() {
  //   this.isLeaving = false;
  //   //check if the car is trying to leave a oneway area
  //   if (!this.is_in_ow) return false;

  //   let is_blocked = true;
  //   this.isLeaving = true;
  //   for (var i = 0; i < this.game.ow_cords.length; i++) {
  //     // use array[i] here
  //     let ow_pos = this.game.ow_cords[i];
  //     if (ow_pos.x === this.goal.x && ow_pos.y === this.goal.y) {
  //       is_blocked = false;
  //       this.isLeaving = false;
  //     }
  //   }
  //   return is_blocked;
  // }

  getDir(vec) {
    if (vec.x > 0) {
      return "right";
    } else if (vec.x < 0) {
      return "left";
    } else if (vec.y > 0) {
      return "down";
    } else if (vec.y < 0) {
      return "up";
    } else {
      return "same";
    }
  }

  find_is_blocking() {
    //check if the car is blocked
    let is_blocked = false;
    for (var i = 0; i < this.game.blocking_cords.length; i++) {
      // use array[i] here
      let blocking_pos = this.game.blocking_cords[i];
      if (blocking_pos.x === this.goal.x && blocking_pos.y === this.goal.y) {
        is_blocked = true;
      }
    }
    return is_blocked;
  }
  findActionIndex(a) {
    //actions = [[-1,0],[1,0],[0,-1],[0,1]]
    let actions = [
      { x: -1, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: 0, y: 1 }
    ];

    for (var i = 0; i < actions.length; i++) {
      if (actions[i].y === a.x && actions[i].x === a.y) {
        return i;
      }
    }
    return false;
  }

  updateScore(a) {
    if (!this.just_spawned) {
      // console.log("updating");
      let aI = this.findActionIndex(a);
      this.updatedScore = true;
      if (this.loadedRFunc) {
        let ds= this.rFunc[this.curStatePrevCords.y][
          this.curStatePrevCords.x
        ][aI];
        this.game.score += ds;

        if (!this.game.reached_terminal) {
          if (!this.game.is_in_ow(this.goal)) {
            this.game.gasScore -= 1;
          } else {
            this.game.gasScore -= 2;
          }
        } else {
          if (ds === 50) {
            this.game.hitFlag = true;
          } else if (ds === -50) {
            this.game.hitPerson = true;

          }
        }

        this.game.pScore = this.game.score - this.game.gasScore;

      }

      // console.log(
      //   this.rFunc[this.curStatePrevCords.y][this.curStatePrevCords.x][aI]
      // );


      let ns = {
        x: this.curStatePrevCords.x + a.x,
        y: this.curStatePrevCords.y + a.y
      };

      if (ns.x >= 0 && ns.x < 10 && ns.y >= 0 && ns.y < 10) {
        if (
          !this.inBlocking(ns)
          // &&
          // !(
          //   this.game.is_in_ow(this.curStatePrevCords) &&
          //   !this.game.is_in_ow(ns)
          // )
        )
          this.curStatePrevCords = ns;
      }
    }
  }

  update(deltaTime) {
    this.game.find_is_leaving_ow(this.goal);

    if (this.game.is_in_ow(this.goal)) {
      if (this.speed.x > 0)this.speed.x = 0.05;
      if (this.speed.x < 0)this.speed.x = -0.05;
      if (this.speed.y > 0)this.speed.y = 0.05;
      if (this.speed.y < 0)this.speed.y = -0.05;
    }


    this.new_pos = {
      x: this.position.x + this.speed.x,
      y: this.position.y + this.speed.y
    };

    if (!this.isdone()) {
      this.position = this.new_pos;
    }

    if (this.isdone()) {
      // if (!this.game.is_in_ow()) {
      if (
        this.goal.x !== this.lastCol.x ||
        this.goal.y !== this.lastCol.y ||
        this.incAction
      ) {
        if (!this.isAnimating) window.timestep += 1;
        if (this.incAction) {
          //this.lastAct
          this.updateScore({
            x: this.lastAct.x,
            y: this.lastAct.y
          });
        } else {
          this.updateScore({
            x: this.goal.x - this.lastCol.x,
            y: this.goal.y - this.lastCol.y
          });
        }
        this.just_spawned = false;
        this.incAction = false;
      }
      if (this.game.reached_terminal) {
        this.lastCol = this.goal;
        this.reachedGoal = true;
        this.stop();
        return;
      }
      if (!this.game.isLeaving) this.lastCol = this.goal;

      this.reachedGoal = true;
      this.stop();
    }
  }
}
