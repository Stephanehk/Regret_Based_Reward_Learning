// // import screenshot from "desktop-screenshot";
// const fs = require('fs')
// const screenshot = require('screenshot-desktop')
// var {spawn} = require('child_process')

export default class TrajHandler {
  constructor(vehicle, game) {
    this.trajLength = 3;
    this.vehicle = vehicle;
    this.game = game;
    this.loadPassingSpaces().then(() => {
      this.loadedTrajData= true;
    });

    // this.json = require("/assets/trajData/ql_passing_spaces.json");
    this.loadedRFunc = false;
    this.tookScreenShot = false;
    this.loadRewardFunction().then(() => {
      this.loadedRFunc = true;
    });

    this.i = 0;
    this.prevI = -1;
    this.cordsI = 0;
    this.quadI = 0;
    this.trajp = 0;

    this.cellSize = vehicle.cellSize;
    this.trajCords = [];
    this.actions = [];

    // this.fillStyles = ["red", "blue", "green", "purple"];

    this.nStyles = 0;
    this.pos2img = {
      left: "img_car_side_f",
      right: "img_car_side",
      up: "img_car_back",
      down: "img_car_front"
    };

    this.pos2dim = {
      left: { x: this.vehicle.sideSizeX, y: this.vehicle.sideSizeY },
      right: { x: this.vehicle.sideSizeX, y: this.vehicle.sideSizeY },
      up: { x: this.vehicle.backSizeX, y: this.vehicle.backSizeY },
      down: { x: this.vehicle.backSizeX, y: this.vehicle.backSizeY }
    };
    this.incompleteCords = [];
    this.incompleteActions = [];
    this.x_img = document.getElementById("img_x");
    this.img_arrow_right = document.getElementById("img_arrow_right");
    this.img_arrow_left = document.getElementById("img_arrow_left");
    this.isPrinted = false;
    this.score = 0;
    this.gasScore = 0;
    this.pScore = 0;
  }

  async loadPassingSpaces() {
    this.json = await (
      await fetch("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/trajData/ql_passing_spaces.json")
      ).json();

    this.keys = [];
    for (var i in this.json) {
      this.keys.push(i);
    }
    // console.log(this.keys);
  }

  async loadRewardFunction() {
    // let rjson = require("/assets/trajData/" +
    //   String(this.game.boardName) +
    //   "_rewards_function.json");
    let rjson = await (
      await fetch('https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/boards/' + String(this.game.boardName) + '_rewards_function.json')
    ).json();
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

  updatePos(ns) {
    this.vehicle.goal = ns;
    if (!this.game.find_is_leaving_ow(ns) && !this.vehicle.find_is_blocking()) {
      if (ns.x >= 0 && ns.x <= 9 && ns.y >= 0 && ns.y <= 9) {
        this.vehicle.lastCol = ns;
        this.vehicle.position = ns;
      }
    }
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
      if (actions[i].x === a.x && actions[i].y === a.y) {
        return i;
      }
    }
    return false;
  }

  updateScore(a) {
    if (this.vehicle.isDone) {
      if (!this.game.reached_terminal) {
        let aI = this.findActionIndex({ x: a[0], y: a[1] });
        if (this.loadedRFunc) {
          this.score += this.rFunc[this.curStatePrevCords.y][
            this.curStatePrevCords.x
          ][aI];
          this.prevI = this.i;

          // console.log(
          //   this.rFunc[this.curStatePrevCords.y][this.curStatePrevCords.x][aI]
          // );
          if (!this.game.is_in_ow(this.curStatePrevCords)) {
            this.gasScore -= 1;
          } else {
            this.gasScore -= 2;
          }
          this.pScore = this.score - this.gasScore;
        }

        // console.log(a);
        // console.log(this.curStatePrevCords);
        // console.log(
        //   this.rFunc[this.curStatePrevCords.y][this.curStatePrevCords.x]
        // );
        // console.log("\n");
        let ns = {
          x: this.curStatePrevCords.x + a[1],
          y: this.curStatePrevCords.y + a[0]
        };
        if (ns.x >= 0 && ns.x < 10 && ns.y >= 0 && ns.y < 10) {
          if (
            !this.inBlocking(ns) &&
            !(
              this.game.is_in_ow(this.curStatePrevCords) &&
              !this.game.is_in_ow(ns)
            )
          )
            this.curStatePrevCords = ns;
        }
      }
    }
  }

  executeTraj(traj) {
    //spawn vehicle at position
    // console.log(this.i);
    if (this.i === 0) {
      let spawnPoint = { x: traj[0][1], y: traj[0][0] };
      this.trajCords.push(spawnPoint);
      // console.log(spawnPoint);
      this.vehicle.position = spawnPoint;
      this.vehicle.goal = spawnPoint;
      this.vehicle.just_spawned = false;
      this.i += 1;
      this.curStatePrevCords = spawnPoint;
      // this.actions.push(null);
    }

    //respawn vehicle back at the beggining
    if (this.i === this.trajLength + 1) {
      let spawnPoint = { x: traj[0][1], y: traj[0][0] };
      this.vehicle.position = spawnPoint;
      this.vehicle.goal = spawnPoint;
      this.vehicle.img = document.getElementById(
        this.pos2img[this.firstAction]
      );
      this.vehicle.size_x = this.pos2dim[this.firstAction].x;
      this.vehicle.size_y = this.pos2dim[this.firstAction].y;

      this.vehicle.just_spawned = true;
      return;
    }

    //execute actions

    let a = traj[this.i];

    //----------------------------------------------------
    if (this.prevI !== this.i) this.updateScore(a);
    // console.log(this.i);

    if (a[0] === 0 && a[1] === 1 && this.vehicle.updatedScore === true) {
      // console.log("right");
      if (this.i === 1) {
        this.firstAction = "right";
      }
      this.vehicle.moveRight();
      if (this.vehicle.isDone) this.actions.push("right");
    }
    if (a[0] === 0 && a[1] === -1 && this.vehicle.updatedScore === true) {
      // console.log("left");
      if (this.i === 1) {
        this.firstAction = "left";
      }
      this.vehicle.moveLeft();
      if (this.vehicle.isDone) this.actions.push("left");
    }

    if (a[0] === 1 && a[1] === 0 && this.vehicle.updatedScore === true) {
      // console.log("down");
      if (this.i === 1) {
        this.firstAction = "down";
      }
      this.vehicle.moveDown();
      // console.log("here");
      if (this.vehicle.isDone) this.actions.push("down");
    }
    if (a[0] === -1 && a[1] === 0 && this.vehicle.updatedScore === true) {
      // console.log("up");
      if (this.i === 1) {
        this.firstAction = "up";
      }
      this.vehicle.moveUp();
      if (this.vehicle.isDone) this.actions.push("up");
    }

    let trajPoint = this.vehicle.goal;
    if (this.inBlocking(trajPoint) === true) {
      if (this.inBlocking(this.vehicle.lastCol)) return;
      trajPoint = this.vehicle.lastCol;

      // console.log(trajPoint);
    }

    let dup = false;
    if (
      this.trajCords[this.trajCords.length - 1].x === trajPoint.x &&
      this.trajCords[this.trajCords.length - 1].y === trajPoint.y
    ) {
      dup = true;
    }
    // console.log(this.incompleteCords);
    // console.log(this.incompleteActions);

    if (dup === false) this.trajCords.push(trajPoint);
  }

  isIncomplete(cords, a) {
    let is_inc = false;
    // console.log(this.incompleteCords.length);
    for (var i = 0; i < this.incompleteCords.length; i++) {
      // use array[i] here
      let blocking_pos = this.incompleteCords[i];
      //&& this.incompleteActions[i] === a
      if (
        blocking_pos.x === cords.x &&
        blocking_pos.y === cords.y &&
        this.incompleteActions[i] === a
      ) {
        is_inc = true;
      }
    }
    return is_inc;
  }

  drawSVG(ctx) {
    this.svgArrow(ctx, 100, 100, 200, 250);

    var svgElement = document.getElementById("SVG");
    let { width, height } = svgElement.getBBox();
    let clonedSvgElement = svgElement.cloneNode(true);
    let outerHTML = clonedSvgElement.outerHTML,
      blob = new Blob([outerHTML], { type: "image/svg+xml;charset=utf-8" });
    let URL = window.URL || window.webkitURL || window;
    let blobURL = URL.createObjectURL(blob);
    let image = new Image();
    // image.onload = () => {
    // let canvas = document.createElement("canvas");
    // canvas.widht = width;

    // canvas.height = height;
    // let context = canvas.getContext("2d");
    // draw image in canvas starting left-0 , top - 0
    image.src = blobURL;
    ctx.drawImage(image, 0, 0, 200, 200);

    //  downloadImage(canvas); need to implement
    //};
  }

  svgArrow(context, p1x, p1y, p2x, p2y) {
    // var p1x = parseFloat(document.getElementById("au").getAttribute("cx"));
    // var p1y = parseFloat(document.getElementById("au").getAttribute("cy"));
    // var p2x = parseFloat(document.getElementById("sl").getAttribute("cx"));
    // var p2y = parseFloat(document.getElementById("sl").getAttribute("cy"));

    // mid-point of line:
    var mpx = (p2x + p1x) * 0.5;
    var mpy = (p2y + p1y) * 0.5;

    // angle of perpendicular to line:
    var theta = Math.atan2(p2y - p1y, p2x - p1x) - Math.PI / 2;

    // distance of control point from mid-point of line:
    var offset = 30;

    // location of control point:
    var c1x = mpx + offset * Math.cos(theta);
    var c1y = mpy + offset * Math.sin(theta);

    // show where the control point is:
    // var c1 = document.getElementById("cp");
    // c1.setAttribute("cx", c1x);
    // c1.setAttribute("cy", c1y);

    // construct the command to draw a quadratic curve
    var curve =
      "M" + p1x + " " + p1y + " Q " + c1x + " " + c1y + " " + p2x + " " + p2y;
    var curveElement = document.getElementById("curve");
    curveElement.setAttribute("d", curve);
  }

  trajBounds() {
    let lowestX = null;
    let lowestY = null;
    let highestX = -1;
    let highestY = -1;
    for (var i = 0; i < this.trajCords.length; i++) {
      let c = this.trajCords[i];

      if (lowestY === null || c.y < lowestY) {
        lowestY = c.y;
      }
      if (c.y > highestY) {
        highestY = c.y;
      }
      if (c.x > highestX) {
        highestX = c.x;
      }
      if (lowestX === null || c.x < lowestX) {
        lowestX = c.x;
      }
    }
    return { lx: lowestX, ly: lowestY, hx: highestX, hy: highestY };
  }

  cord2canvas(val, size) {
    return this.cellSize * val + this.cellSize / 2 - size / 2;
  }

  maskOutBg(ctx) {
    let rectCords = this.trajBounds();
    //overlay
    //lighten
    ctx.globalCompositeOperation = "saturation";
    ctx.beginPath();

    let box1h;
    if (rectCords.lx === 0) {
      box1h = this.cord2canvas(rectCords.ly - 0.5, 0);
    } else {
      box1h = this.game.gameHeight;
    }
    //left rect
    ctx.rect(0, 0, this.cord2canvas(rectCords.lx - 0.5, 0), box1h);
    //bottom rect
    ctx.rect(
      0,
      this.cord2canvas(rectCords.hy + 0.5, 0),
      this.game.gameWidth,
      this.game.gameHeight
    );
    //right rect
    let box3h;
    if (rectCords.hx === 9) {
      box3h = this.cord2canvas(rectCords.ly + 0.5, 0);
    } else {
      box3h = this.game.gameHeight;
    }
    ctx.rect(
      this.cord2canvas(rectCords.hx + 0.5, 0),
      0,
      this.game.gameWidth,
      box3h
    );
    let box4h;
    if (rectCords.ly === 0) {
      box4h = 0;
    } else {
      box4h = this.cord2canvas(rectCords.ly + 0.5, 0);
    }
    //top rect
    ctx.rect(
      this.cord2canvas(rectCords.lx - 1 + 0.5, 0),
      0,
      this.game.gameWidth,
      box4h
    );
    ctx.fillStyle = "whitesmoke";
    ctx.fill();
  }

  recolorCanvas(ctx) {
    ctx.globalAlpha = 0.15;
    ctx.globalCompositeOperation = "saturation";
    // ctx.beginPath();
    // ctx.rect(0, 0, this.game.gameWidth, this.game.gameHeight);
    // ctx.fillStyle = "whitesmoke";
    // ctx.fill();
  }

  handelSameDir(p1, p2, ctx, first = false) {
    if (
      p1 === undefined ||
      p2 === undefined ||
      p1.inc === true ||
      p2.inc === true
    )
      return;
    if (
      (p1.a === "left" && p2.a === "right") ||
      (p1.a === "right" && p2.a === "left")
    ) {
      if (first) this.sameDirYoffset = -8;
      else this.sameDirYoffset = 8;
    } else if (
      (p1.a === "up" && p2.a === "down") ||
      (p1.a === "down" && p2.a === "up")
    ) {
      // console.log("here");
      if (first) this.sameDirXoffset = -8;
      else this.sameDirXoffset = 8;
    }
  }

  handelMulSameDir(p1, p2, p3, ctx, first = false) {
    if (
      ((p1.a === "left" && p2.a === "right") ||
        (p1.a === "right" && p2.a === "left")) &&
      (p3.a === "left" || p3.a === "right")
    ) {
      this.sameDirYoffset = 0;
    } else if (
      ((p1.a === "up" && p2.a === "down") ||
        (p1.a === "down" && p2.a === "up")) &&
      (p3.a === "up" || p3.a === "down")
    ) {
      // console.log("here");
      this.sameDirXoffset = 0;
    }
  }

  canvasArrow(ctx, p1, p2, p3 = null, p4 = null) {
    this.sameDirXoffset = 0;
    this.sameDirYoffset = 0;
    let x;
    let y;
    if (p1 != null) {
      x = this.cord2canvas(p1.x, 0);
      y = this.cord2canvas(p1.y, 0);
    }

    let xn = this.cord2canvas(p2.x, 0);
    let yn = this.cord2canvas(p2.y, 0);

    ctx.beginPath();
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 5;
    // console.log("here");
    if (p1 != null && p1.inc === true && p2.inc === true) {
      return;
    }

    if (p3 != null) {
      this.handelSameDir(p1, p3, ctx);
      //if prev is incomplete, compare current to prev prev
      if (this.lastInc === true) this.handelSameDir(p2, p3, ctx);
    }

    if (p1 != null) {
      this.handelSameDir(p1, p2, ctx, true);
      //if current is incomplete, compare prev to next
      if (p4 !== null && p2.inc === true) this.handelSameDir(p1, p4, ctx, true);
    }

    if (p1 != null && p3 != null) this.handelMulSameDir(p1, p2, p3, ctx);

    if (p1 != null && p1.inc && !p2.inc) {
      // console.log("here");
      //previous point is incomplete
      // console.log("here");
      if (p2.y - 1 === p1.y) {
        // console.log("here");
        ctx.moveTo(x + this.sameDirXoffset, y + 20 + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn + this.sameDirYoffset);
      }

      if (p1.y - 1 === p2.y) {
        ctx.moveTo(x + this.sameDirXoffset, y + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn + this.sameDirYoffset);
      }
      //moving from left to rigjt
      if (p2.x - 1 === p1.x) {
        // console.log("here");
        ctx.moveTo(x + 5 + this.sameDirXoffset, y - 2 + this.sameDirYoffset);
        ctx.lineTo(xn - 20 + this.sameDirXoffset, yn - 2 + this.sameDirYoffset);
      }
      //moving from right to left
      if (p1.x - 1 === p2.x) {
        // console.log("here");
        // ctx.moveTo(x, y - 10);
        // ctx.lineTo(x, y - 2);
        ctx.moveTo(x + this.sameDirXoffset, y - 2 + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn - 2 + this.sameDirYoffset);
      }
    } else if (p1 != null && p2.inc && !p1.inc) {
      // console.log("here");

      if (p2.x - 1 === p1.x) {
        ctx.moveTo(x + this.sameDirXoffset, y - 2 + this.sameDirYoffset);
        ctx.lineTo(xn + 5 + this.sameDirXoffset, yn - 2 + this.sameDirYoffset);
      }

      if (p2.y - 1 === p1.y) {
        // console.log("here");
        ctx.moveTo(x + this.sameDirXoffset, y - 5 + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn + this.sameDirYoffset);
      }

      if (p1.y - 1 === p2.y) {
        // console.log("here");
        ctx.moveTo(x + this.sameDirXoffset, y - 5 + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn + this.sameDirYoffset);
      }

      //moving from right to left
      if (p1.x - 1 === p2.x) {
        // console.log("here");
        ctx.moveTo(x - 5 + this.sameDirXoffset, y - 2 + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn - 2 + this.sameDirYoffset);
      }
      // console.log("here");
    } else {
      //check for prevprev (s,a) pair

      // if (p3 != null && p3.inc === false) {
      //   this.handelSameDir(p2, p3, ctx);
      // }

      // if (p3 != null && p3.inc === false) {
      //   this.handelSameDir(p1, p3, ctx);
      // }

      // if (p1 != null && p1.inc === false) {
      //   this.handelSameDir(p1, p2, ctx, true);
      // }

      if (p1 != null && p1.x !== p2.x) {
        // console.log("here");
        // console.log(p1.a);
        // console.log(p2.a);
        // console.log("\n");
        ctx.moveTo(x + this.sameDirXoffset, y - 3 + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn - 3 + this.sameDirYoffset);
      } else if (p1 != null && p1.y !== p2.y) {
        // console.log("here");
        ctx.moveTo(x + this.sameDirXoffset, y + this.sameDirYoffset);
        ctx.lineTo(xn + this.sameDirXoffset, yn + this.sameDirYoffset);
      }
    }
    ctx.stroke();
  }

  drawTriangle(ctx, tipCordX, tipCordY, xDir, yDir) {
    ctx.beginPath();

    let offset1 = 0;
    let offset2 = 0;
    let offset3 = 0;
    let offset4 = 0;
    let offset5 = 0;
    let offset6 = 0;
    let offset7 = 0;
    let label_offset_x = 0;
    let label_offset_y = 0;

    if (xDir > 0) {
      offset1 = 15;
      offset2 = -30;
      offset3 = -10;
      label_offset_x = -12;
      label_offset_y = 2;
    } else if (xDir < 0) {
      offset1 = 0;
      offset2 = 0;
      offset3 = 10;
      label_offset_x = 8;
      label_offset_y = 2;
    } else if (yDir > 0) {
      // console.log("here");
      offset2 = -3.5;
      offset3 = 10;
      offset4 = 10 - 10;
      offset5 = -24;
      offset6 = -4;
      offset7 = -25;
      label_offset_x = -2;
      label_offset_y = -7;
    } else if (yDir < 0) {
      // console.log("here");
      offset2 = -3.5;
      offset3 = 10;
      offset4 = -5 + 10;
      offset5 = -24;
      offset6 = 24;
      offset7 = 5;
      label_offset_x = -2;
      label_offset_y = 11;
    }
    // ctx.fillStyle = prevFill;
    let prevFill = ctx.fillStyle.valueOf();

    tipCordY = tipCordY + 5 + offset4 + this.sameDirYoffset;
    tipCordX = tipCordX + offset3 + offset1 + this.sameDirXoffset;
    let arrowLength = 15;
    let arrowHeight = 20;
    ctx.moveTo(tipCordX, tipCordY);
    ctx.lineTo(
      tipCordX + arrowLength + offset2,
      offset7 + tipCordY + arrowHeight / 2
    );
    ctx.lineTo(
      tipCordX + arrowLength + offset2 + offset5,
      offset6 + tipCordY - arrowHeight / 2
    );

    ctx.fill();
    ctx.font = "9px CustomFont";
    ctx.fillStyle = "white";
    let text;

    if (prevFill === "#c0a948") text = "1";
    else if (prevFill === "#c0aac2") text = "2";
    else text = 3;

    // console.log(text);

    ctx.fillText(text, tipCordX + label_offset_x, tipCordY + label_offset_y);
    ctx.fillStyle = prevFill;
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

  remove(arr, cords) {
    let ret = [];
    for (var i = 0; i < arr.length; i++) {
      if (
        !(
          arr[i].x === cords.x &&
          arr[i].y === cords.y &&
          arr[i].inc === cords.inc
        )
      ) {
        ret.push(arr[i]);
      }
    }
    return ret;
  }

  drawX(ctx, x, y) {
    let size = 6;

    ctx.setLineDash([]);
    ctx.lineWidth = 5;
    ctx.beginPath();

    ctx.moveTo(x - size + 10, y - size + 5);
    ctx.lineTo(x + size + 10, y + size + 5);

    ctx.moveTo(x + size + 10, y - size + 5);
    ctx.lineTo(x - size + 10, y + size + 5);
    ctx.stroke();

    ctx.setLineDash([5, 5]);
  }

  updateColors(ctx, replace = false, increment = 1) {
    // console.log(this.fillStyles[this.nStyles]);

    while (this.styleStatus[this.nStyles] === true) {
      this.nStyles += 1;
    }

    ctx.fillStyle = this.fillStyles[this.nStyles];
    ctx.strokeStyle = this.fillStyles[this.nStyles];
    this.styleStatus[this.nStyles] = true;
    // if (replace) {
    //   this.fillStyles[this.nStyles] = this.fillStyles[this.nStyles + 1];
    // }
    this.nStyles += increment;
  }

  draw(ctx) {
    // this.maskOutBg(ctx);
    ctx.globalAlpha = 1;

    let centers = [];
    // console.log(this.trajCords.length);
    this.nStyles = 0;
    // this.updateColors(ctx);
    let nMoreCols = 3 - this.trajCords.length;
    // this.fillStyles = ["darkgreen", "darkseagreen", "lightcoral", "indianred"];
    this.fillStyles = [
      "rgb(192, 169, 72)",
      "rgb(192, 170, 194)",
      "rgb(127, 117, 251)"
    ];

    this.styleStatus = [false, false, false, false];

    //find centers of trajectory
    // console.log(this.trajCords);
    for (var n = 0; n < this.trajCords.length; n++) {
      let cords = this.trajCords[n];
      // console.log(cords);
      // let cordsX = this.cord2canvas(cords.x, 20);
      // let cordsY = this.cord2canvas(cords.y, 15);
      // let drawArrow = false;
      // let inc = false;
      let a = this.actions[n];
      if (this.isIncomplete(cords, a)) {
        centers.push({ x: cords.x, y: cords.y, inc: true, a: a });
        if (n !== 0) {
          //if cords not in blocking or ow, it must have been one way
          if (!this.inBlocking(cords) && !this.game.find_is_leaving_ow(cords)) {
            centers.push({
              x: this.trajCords[n].x,
              y: this.trajCords[n].y,
              inc: false,
              a: a
            });
          } else {
            centers.push({
              x: this.trajCords[n - 1].x,
              y: this.trajCords[n - 1].y,
              inc: false,
              a: a
            });
          }

          // drawArrow = true;
        }
      } else {
        // console.log(a);
        // console.log(n);
        // console.log("\n");

        centers.push({ x: cords.x, y: cords.y, inc: false, a: a });
      }
    }

    // console.log(centers);
    // console.log(this.trajCords);
    for (var i = 0; i < this.trajCords.length; i++) {
      // console.log(cords);
      // this.updateColors(ctx);
      let cords = this.trajCords[i];
      // console.log(cords);
      let cordsX = this.cord2canvas(cords.x, 20);
      let cordsY = this.cord2canvas(cords.y, 15);
      let drawArrow = false;
      let inc = false;
      let a = this.actions[i];
      let blockingCase = false;
      // console.log(a);
      // console.log(i);
      // console.log("\n");
      // console.log(this.actions.length);
      // console.log(this.isIncomplete(this.trajCords[2]));
      if (this.isIncomplete(cords, a)) {
        blockingCase = true;
        // console.log(a);
        inc = true;
        let offsetY;
        let offsetX;

        if (a === "up") {
          offsetX = 0;
          if (this.inBlocking(cords)) offsetY = 20;
          else offsetY = -20;
        } else if (a === "down") {
          // console.log("here");
          offsetX = 0;
          //idk if this is right
          if (this.inBlocking(cords)) offsetY = -20;
          else offsetY = 20;
        } else if (a === "left") {
          offsetX = -20;
          offsetY = -2;
        } else if (a === "right") {
          offsetX = 25;
          offsetY = 0;
        }

        // console.log(cords);
        if (i === 0) {
          this.updateColors(ctx, true, 1);
        } else {
          this.nStyles += 1;
          this.updateColors(ctx, true, -1);
        }

        this.drawX(ctx, cordsX + offsetX, cordsY + offsetY);
        drawArrow = true;
        for (var nCol = 0; nCol < nMoreCols; nCol++) {
          if (i === 0) {
            this.updateColors(ctx, true, 1);
          } else {
            this.nStyles += nCol + 2;
            this.updateColors(ctx, true, -(nCol + 2));
          }

          this.drawX(ctx, cordsX + offsetX + 7 * (nCol + 1), cordsY + offsetY);
        }
      }
      // else {
      //   if (i !== 0) {
      //     // console.log(cords);
      //     drawArrow = true;
      //   }
      // }

      let currentCords = centers[i];

      if (i > 0) {
        // console.log(centers[i].a);
        // console.log(i);
        // console.log("\n");
        let x = -1;
        let n = 1;
        let prevCords = centers[i - 1];
        this.updateColors(ctx);
        if (i > 1) {
          // console.log(currentCords);
          if (prevCords.inc === true) x = -2;
          //add another negative one to prevprev if currentCords.inc === true
          this.canvasArrow(ctx, prevCords, currentCords, centers[i - 2], null);
        } else {
          //this handles the case where we have left-> up (collision) ->right
          if (currentCords.inc === true) n = 2;
          this.canvasArrow(ctx, prevCords, currentCords, null, centers[i + n]);
        }
        // console.log(cords);

        this.drawTriangle(
          ctx,
          cordsX,
          cordsY,
          currentCords.x - prevCords.x,
          currentCords.y - prevCords.y
        );

        if (currentCords.inc === true) {
          centers = this.remove(centers, currentCords);
          this.lastInc = true; //this flag needs to be here becuase we remove the incomplete points
        } else {
          this.lastInc = false;
        }
      }
    }

    this.recolorCanvas(ctx);
  }

  // saveBase64AsFile(base64, fileName) {
  //   var link = document.createElement("a");
  //   document.body.appendChild(link);
  //   link.setAttribute("type", "hidden");
  //   link.href = "data:text/plain;base64," + base64;
  //   link.download = fileName;
  //   link.click();
  //   document.body.removeChild(link);
  // }

  update(deltaTime) {
    //NOTE!! FOR DEBUGGING PURPOSES - DONT FORGET
    if (this.quadI === 0) {
      this.cordsI = 0;
      this.quadI = 0;
      this.trajp = 0;
      // if (!this.tookScreenShot) {
      //   this.cur_name = "/trajImages/" +
      //     String(this.keys[this.cordsI]) +
      //     "_" +
      //     String(this.quadI) +
      //     "_" +
      //     String(this.trajp);
      //   screenshot({format: 'png'}).then((img) => {
      //     // img: Buffer filled with jpg goodness
      //     fs.writeFile('out.jpg', img, function (err) {
      //       if (err) {
      //         throw err
      //       }
      //       console.log('written to out.jpg')
      //     })
      //     // ...
      //   }).catch((err) => {
      //     console.log(err);
      //   })
      //
      //   this.tookScreenShot = true;
      // }

    }

    if (this.i < this.trajLength + 2 && this.loadedRFunc && this.loadedTrajData) {
      let traj1 = this.json[this.keys[this.cordsI]][this.quadI][this.trajp];

      this.executeTraj(traj1);
    }

    if (
      this.vehicle.isDone === true &&
      this.vehicle.updatedScore === true &&
      this.loadedRFunc && this.loadedTrajData
    ) {
      // if (this.i === this.trajLength + 2) {
      //   this.tookScreenShot = false;
      // }
      if (this.i < this.trajLength + 2) {
        this.i += 1;

      }
      // else if (this.trajp === 0) {
      //   this.trajp = 1;
      //   this.i = 0;
      //   this.trajCords = []
      // } else if (this.quadI < this.json[this.keys[this.cordsI]].length - 1) {
      //   this.quadI += 1;
      //   this.trajp = 0;
      //   this.i = 0;
      // } else if (this.cordsI < this.json.length - 1) {
      //   this.cordsI += 1;
      //   this.quadI = 0;
      //   this.trajp = 0;
      //   this.i = 0;
      // }
    }
  }
}
