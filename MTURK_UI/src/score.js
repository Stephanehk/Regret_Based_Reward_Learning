export default class Score {
  constructor(game, scoreWidth) {
    this.game = game;
  }
  start() {
    this.score = 0;
    this.prevScore = 0;
    this.prevprevScore = 0;
    this.lastx = 0;
    this.lasty = 0;
    this.lastw = 0;
    this.lasth = 0;
    this.startY = 300;
    this.decomposedScore = false;
    this.printDecomposed = true;
    this.showRects = false;
    this.gasScore = 0;
    this.prevGasScore = 0;
    this.pScore = 0;
    this.prevPScore = 0;
    this.maxHeight = 500;
    this.updateAll = false;
    this.loadValueFunction().then(() => {
      this.loadedVFunc = true;
    });
  }

  async loadValueFunction() {
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

    let vjson = await (
      await fetch('https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/boards/' + String(this.game.boardName) + '_value_function.json')
    ).json();
    //console.log(json)

    // console.log(rjson);
    this.vFunc = [];
    for (var i in vjson) {
      let row = [];
      for (var j in vjson[i]) {
        row.push(vjson[i][j]);
      }
      this.vFunc.push(row);
      // this.keys.push(i);
    }
  }

  drawRect(ctx, x, y, w, h, color) {
    if (color === "white")console.log("printing white rect!!");
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
    ctx.stroke();
  }

  barText(ctx, val, y_offset, text = "", fontSize = "20px") {
    ctx.font = fontSize + " CustomFont";
    let symbol = "$";
    if (val > 0) {
      ctx.fillStyle = "green";
      symbol = "+ $";
    }
    if (val < 0) {
      ctx.fillStyle = "red";
      symbol = "- $";
      val = -val;
    }
    if (val === 0) ctx.fillStyle = "brown";
    if (this.lasty > 20)
      ctx.fillText(
        text + symbol + String(val),
        220,
        this.cap(this.lasty + y_offset)
      );
    else
      ctx.fillText(
        text + symbol + String(val),
        220,
        this.cap(this.startY + 10 + y_offset)
      );
  }


  staticBarText(ctx, val, y,x, text = "", fontSize = "20px",keepBlack=false) {
    ctx.font = fontSize + " CustomFont";
    let symbol = "$";
    if (val > 0) {
      ctx.fillStyle = "rgb(83, 117, 76)";
      symbol = "+ $";
    }
    if (val < 0) {
      ctx.fillStyle = "rgb(196, 57, 57)";
      symbol = "- $";
      val = -val;
    }
    if (val === 0) ctx.fillStyle = "brown";
    if (keepBlack)ctx.fillStyle = "black";
    if (this.lasty > 20)
      ctx.fillText(
        text + symbol + String(val),
        x,
        y
      );
    else
      ctx.fillText(
        text + symbol + String(val),
        x,
        y
      );
  }

  barImg(ctx, id, y_offset, x = 180, size = 30) {
    let img = document.getElementById(id);

    if (this.lasty > 20)
      ctx.drawImage(img, x, this.cap(this.lasty + y_offset), size, size);
    else
      ctx.drawImage(img, x, this.cap(this.startY + 10 + y_offset), size, size);
  }

  staticBarImg(ctx, id, y, x, sizex = 30, sizey = 30) {
    let img = document.getElementById(id);
    ctx.drawImage(img, x,y, sizex, sizey);
  }

  drawGasTank(ctx, x) {
    if (this.decomposedScore === false) return;
    let y = this.maxHeight;
    let w = 120;
    let h = -5 * 100;
    this.drawRect(ctx, x, y, w, h, "darkolivegreen");

    y = y - 5 * 100;
    h = -5 * this.gasScore;
    this.drawRect(ctx, x, y, w, h, "grey");

    ctx.font = "20px CustomFont";
    ctx.fillStyle = "black";
    // console.log("here");
    ctx.fillText(
      "Gas: $" + String(this.gasScore),
      x + 10,
      550 + 10 - 5 * this.gasScore - 450
    );
  }

  cap(y) {
    // return y;
    if (y > 500 || this.updateAll) {
      this.updateAll = true;
      let out = y - 500;

      // console.log(y);
      // console.log(500 + (y - 500));
      // console.log("\n");
      return 400 + out;
    }
    return y;
    // let max = 300;
    // if (Math.abs(val) + y > max) {
    //   if (val < 0) {
    //     val = -(max - y);
    //   } else {
    //     val = max - y;
    //   }
    // }
  }

  drawRects(ctx, x, w) {
    if (this.prevScore < 0) {
      // console.log(this.score);
      // console.log(this.prevscore);
      let py = this.startY - 5 * this.prevScore + 3;

      let ph = 5 * this.prevScore;
      this.drawRect(ctx, x, py, w, ph, "brown");

      if (this.delta < 0) {
        let y = py - 5 * (this.score - this.prevScore);
        let h = 5 * this.delta;
        this.drawRect(ctx, x, y, w, h, "red");

        this.lastx = x;
        this.lasty = y.valueOf();
        this.lastw = w;
        this.lasth = h.valueOf();
        this.color2draw = "red";
        // this.subtracter = 0;
        // this.step = 0;
      } else if (this.delta > 0) {
        let y;
        let h;
        if (this.score > 0) {
          //jump over the bar
          y = this.startY;
          h = -5 * (this.delta - this.prevScore);
          ctx.clearRect(x, py, w, ph);
        } else {
          y = this.startY - 5 * this.prevScore + 3;
          h = -5 * this.delta;
        }
        this.drawRect(ctx, x, y, w, h, "green");

        this.lastx = x;
        this.lasty = y.valueOf();
        this.lastw = w;
        this.lasth = h.valueOf();
        this.color2draw = "green";
      }
    }

    if (this.prevScore > 0) {
      let ph = -5 * this.prevScore;
      this.drawRect(ctx, x, this.startY, w, ph, "brown");

      if (this.delta > 0) {
        let y = this.startY + -5 * this.prevScore;
        let h = -5 * this.delta;
        this.drawRect(ctx, x, y, w, h, "green");
        // console.log(h);
        // console.log(y);

        this.lastx = x;
        this.lasty = y.valueOf();
        this.lastw = 120;
        this.lasth = h.valueOf();
        this.color2draw = "green";
      }
      if (this.delta < 0) {
        let y;
        let h;
        if (this.score < 0) {
          y = this.startY + 3;
          h = -5 * (this.delta - this.prevScore);
          ctx.clearRect(x, this.startY, w, ph);
        } else {
          y = this.startY + -5 * this.prevScore;
          h = -5 * this.delta;
        }
        this.drawRect(ctx, x, y, w, h, "red");

        this.lastx = x;
        this.lasty = y.valueOf();
        this.lastw = 120;
        this.lasth = h.valueOf();
        this.color2draw = "red";
      }
    }

    //show newest part of score
    this.drawRect(
      ctx,
      this.lastx,
      this.lasty,
      this.lastw,
      this.lasth,
      this.color2draw
    );
  }

  setupLabels(ctx, x, w) {
    if (this.game.dispTraj === false) {
      if (this.showRects)this.drawRect(ctx, x - 10, this.startY, 160, 3, "black");
    }
    ctx.font = "30px CustomFont";

    if (this.printDecomposed === true) {
      if (this.game.dispTraj === false) {


        if (this.showRects) {
          ctx.font = "25px CustomFont";
          ctx.fillStyle = "black";
          ctx.fillText("Score: $" + String(this.score), 10, 50);
          this.barImg(ctx, "img_gas", -25);
          this.barText(ctx, this.gasScore - this.prevGasScore, 0);
          this.barImg(ctx, "img_total", 10);
          this.barText(ctx, this.pScore - this.prevPScore, 30);
        } else {

          // if (window.observationType === 0) {
          ctx.font = "25px CustomFont";
          ctx.fillStyle = "black";
          ctx.fillText("Score: $" + String(this.score), 10, 50);
          this.staticBarImg(ctx, "img_gas", 70,120,50,60)
          this.staticBarText(ctx, this.gasScore, 100,50,"","25px")

          this.staticBarImg(ctx, "img_coin_multiple", 130,120,60,50);
          this.staticBarText(ctx, this.game.nCoins, 160,50,"","25px");

          this.staticBarImg(ctx, "img_garbage_multiple", 190,120,70,50);
          this.staticBarText(ctx, -this.game.nGarbage, 220,50,"","25px");

          this.staticBarImg(ctx, "img_person", 250,120,50,60);
          this.staticBarText(ctx, -50*this.game.nPeople, 290,50,"","25px");

          this.staticBarImg(ctx, "img_flag", 320,120,60,60);
          this.staticBarText(ctx, 50*this.game.nFlags, 350,50,"","25px");
          // } else {
          ctx.fillStyle = "rgb(62, 63, 64)";
          ctx.font = "20px CustomFont";
          ctx.fillText(
            "Best Possible",
            5,
            410
          );
          ctx.fillText(
            "Score From Start",
            5,
            435
          );

          if (this.loadedVFunc) {
            // let sv= this.vFunc[this.game.vehicle.curStatePrevCords.y][this.game.vehicle.curStatePrevCords.x]
            let sv= this.vFunc[this.game.spawnPoint.y][this.game.spawnPoint.x]
            this.staticBarText(ctx, sv, 465,10,"","20px")
          }
          ctx.fillStyle = "rgb(62, 63, 64)";
          ctx.fillText(
            "Best Possible Score",
            5,
            515
          );
          ctx.fillText(
            "Given Your Moves",
            5,
            545
          );

          if (this.loadedVFunc) {
            let sv= this.vFunc[this.game.vehicle.curStatePrevCords.y][this.game.vehicle.curStatePrevCords.x]
            // let sv= this.vFunc[this.game.spawnPoint.y][this.game.spawnPoint.x]
            this.staticBarText(ctx, this.score + sv, 575,10,"","20px")
          }

          ctx.fillStyle = "black";
          ctx.font = "25px CustomFont";
          ctx.fillText(
            "Opportunity Cost",
            5,
            620
          );

          if (this.loadedVFunc) {
            let dsv= this.vFunc[this.game.vehicle.curStatePrevCords.y][this.game.vehicle.curStatePrevCords.x] - this.vFunc[this.game.spawnPoint.y][this.game.spawnPoint.x]
            // let sv= this.vFunc[this.game.spawnPoint.y][this.game.spawnPoint.x]
            this.staticBarText(ctx,this.score +  dsv, 650,10,"","25px")
          }

          //}


        }

      } else {
        ctx.font = "40px CustomFont";
        ctx.fillStyle = "black";
        ctx.fillText("Score", 10, 50);
        ctx.fillText("Change: $" + String(this.score), 10, 90);

        if (this.showRects)this.barImg(ctx, "img_gas", -170, 130, 50);
        if (this.showRects)this.barText(ctx, this.gasScore, -140, "", "30px");

        if (this.showRects)this.barImg(ctx, "img_total", -100, 130, 50);
        if (this.showRects)this.barText(ctx, this.pScore, -70, "", "30px");
      }
    } else if (this.decomposedScore === true) {
      ctx.fillStyle = "brown";
      ctx.fillText("Total: $" + String(this.score), 10, 50);
      this.barText(ctx, this.delta, 0);
    } else {
      ctx.fillStyle = "brown";
      ctx.fillText("Score: $" + String(this.score), 10, 50);
      this.barText(ctx, this.delta, 0);
    }
  }

  draw(ctx) {
    this.drawGasTank(ctx, 200);
    let x = 10;
    let w = 140;
    // ctx.fillText("Score: $" + String(this.delta), 10, 100);
    //draw prevscore
    if (this.game.dispTraj === false) {
      if (this.showRects)this.drawRects(ctx, x, w);
      ctx.clearRect(10, 60, w, -100);
      // ctx.clearRect(10, 600, w, -20);
      ctx.fill();

      this.setupLabels(ctx, x, w);
    } else {
      this.setupLabels(ctx, x, w);
    }
  }

  update() {
    if (
      this.game.vehicle.reachedGoal &&
      this.game.vehicle.goal !== this.goalDrawn
    ) {
      if (this.decomposedScore === true) {
        this.prevScore = this.pScore;
        this.score = this.game.pScore;
      } else {
        this.prevScore = this.score;
        this.score = this.game.score;
      }

      this.prevGasScore = this.gasScore;
      this.gasScore = this.game.gasScore;

      this.prevPScore = this.pScore;
      this.pScore = this.game.pScore;

      this.goalDrawn = this.game.vehicle.goal;
      this.delta = this.score - this.prevScore;
      this.lasth = 0;
    }
  }
}
