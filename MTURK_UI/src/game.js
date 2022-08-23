import Vehicle from "/src/vehicle";
import House from "/src/house";
import Flag from "/src/flag";
import Person from "/src/person";
import Coin from "/src/coin";
import Garbage from "/src/garbage";
import OWTile from "/src/owTile";

import InputHandler from "/src/input";
import TrajHandler from "/src/trajHandler";


export default class Game {
  constructor(gameWidth, gameHeight, spawnPoint, boardName, dispTraj) {
    this.gameWidth = gameWidth;
    this.gameHeight = gameHeight;
    this.spawnPoint = spawnPoint;
    this.boardName = boardName;
    // this.is_in_blocking = false;
    this.reached_terminal = false;
    this.nSteps = 0;
    this.collectedCoin = false;
    this.dispTraj = dispTraj;
    this.blockingCords = [];
    this.isLeavingBlank = false;


    //metrics used to see if the user completing our assigned goals
    this.hitPerson = false
    this.hitFlag = false
    this.hitHouse = false
    this.hitEdge = false
    this.triedLeaving = false
    //metrics used to decompose score
    this.nCoins = 0;
    this.nPeople = 0;
    this.nGarbage = 0;
    this.nFlags = 0;
  }

//   fetchJSONFile(path, callback) {
//     var httpRequest = new XMLHttpRequest();
//     httpRequest.onreadystatechange = function() {
//         if (httpRequest.readyState === 4) {
//             if (httpRequest.status === 200) {
//                 var data = JSON.parse(httpRequest.responseText);
//                 if (callback) callback(data);
//             }
//         }
//     };
//     httpRequest.open('GET', path);
//     httpRequest.send();
// }

  start(ins = "",dispIns = true, loadTerm = true, isAnimating = false, id = 0) {
    // if (window.n_games === 7)return;
    this.gameObjects = null;
    this.isAnimating = isAnimating;
    this.animationId = id
    this.vehicle = new Vehicle(this, isAnimating);
    // this.coin = new Coin(this, { x: 0, y: 0 });
    this.score = 0;
    this.gasScore = 0;
    this.pScore = 0;
    this.finishedAnimation = false;
    if (this.dispTraj === false) new InputHandler(this.vehicle);
    // this.board_objects = this.load_board();
    // this.fetchJSONFile(("/assets/boards/" + String(this.boardName) + "_board.json", function(data){
    // // do something with your data
    // console.log(data);
    // });
    // var jsonData = JSON.parse(document.getElementById('data').textContent)
    // console.log(jsonData);


    // Assume the async call always succeed
    this.load_board(loadTerm).then(() => {
      this.gameObjects = [...this.board_objects, this.vehicle];
    });
    this.trajHandler = new TrajHandler(this.vehicle, this);

    if (window.n_games !== 10 && dispIns && !window.im.showMidGameIns) {
      // document.getElementById("displayInsBtn").style.display = "block";
      // displayInsBtn
      $(".displayInsBtn").show();

      $("#myModal").on('show.bs.modal', function (e) {
         var modal = $(this)

         // var mimg = $('.img-responsive');
         // mimg.css("width", 380);
         // mimg.css("height", 110);

         // modal.find('.label1').text(ins)
         // modal.find('.modal_img1').attr("src","assets/images/img_flag.png")
         $(".img-responsive").attr('src', ins);
         //modal_img1

      });
      $("#myModal").modal("show");
    }

  }

  update(deltaTime) {
    this.collectedCoin = false;

    if (this.dispTraj === true) this.trajHandler.update(deltaTime);

    this.gameObjects.forEach((object) => object.update(deltaTime));

    //update oneway gates
    if (this.is_in_ow(this.vehicle.lastCol) || this.is_in_ow(this.spawnPoint)) {
      this.ow_obs.forEach(
        (object) =>
          (object.blockade = document.getElementById("img_blockade_closed"))
      );
    }

    if (this.dispTraj === true) {
      this.score = this.trajHandler.score;
      this.pScore = this.trajHandler.pScore;
      this.gasScore = this.trajHandler.gasScore;
    }
  }

  draw(ctx) {
    ctx.globalCompositeOperation = "source-over";
    this.gameObjects.forEach((object) => object.draw(ctx));
    if (this.dispTraj === true) this.trajHandler.draw(ctx);
    if (this.reached_terminal && this.isAnimating && this.animationId === 1) {
      ctx.font = "30px CustomFont";
      ctx.fillStyle = "green";
      ctx.fillText("+ $50", 20, 525);
      this.finishedAnimation = true;
      // if (window.alpha > 0) {
      //   window.alpha-=window.delta
      //   ctx.globalAlpha = window.alpha;
      // }

    } else if (this.reached_terminal && this.isAnimating && this.animationId === 2) {
      ctx.font = "30px CustomFont";
      ctx.fillStyle = "red";
      ctx.fillText("- $50", 310, 340);
      this.finishedAnimation = true;
      // if (window.alpha > 0) {
      //   window.alpha-=window.delta
      //   ctx.globalAlpha = window.alpha;
      // }
    }
  }

  is_in_ow(cords) {
    // console.log(cords);
    for (var i = 0; i < this.ow_cords.length; i++) {
      let ow_pos = this.ow_cords[i];

      if (ow_pos.x === cords.x && ow_pos.y === cords.y) {
        return true;
      }
    }
    return false;
  }

  find_is_leaving_blank(nextCords) {
    if (!this.is_in_ow(this.vehicle.lastCol) && this.is_in_ow(nextCords)) {
      this.isLeavingBlank = true;
    } else {
      // console.log(this.vehicle.lastCol);
      this.isLeavingBlank = false;
    }
  }

  find_is_leaving_ow(nextCords) {
    return false;
    // //check if the car is trying to leave a oneway area
    // if (!this.is_in_ow(nextCords) && this.is_in_ow(this.vehicle.lastCol)) {
    //   this.isLeaving = true;
    // } else {
    //   // console.log(this.vehicle.lastCol);
    //   this.isLeaving = false;
    // }
  }

  async load_board(loadTerm) {
    let objects = [];

    //https://api.jsonbin.io/b/60ca6d8f8ea8ec25bd0e7835
  //
  //"/assets/boards/" + String(this.boardName) + "_board.json"
    let json = await (
      await fetch('https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/boards/' + String(this.boardName) + '_board.json')
    ).json();
    // console.log(json)

    // fetch('https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/boards/' + String(this.boardName) + '.json')
    // .then(response => response.json())
    // .then(data =>  this.json = data);
    // var json = require("/assets/boards/test_json_board.json");
    var board = [];
    // var blocking_cords = [];
    this.blocking_cords = [];
    this.terminal_cords = [];
    this.ow_cords = [];
    this.ow_obs = [];
    //loads json
    for (var i in json) {
      var row = [];
      for (var j in json[i]) {
        row.push(json[i][j]);
      }
      board.push(row);
    }
    board.forEach((row, rowIndex) => {
      row.forEach((item, itemIndex) => {
        let position = {
          x: itemIndex,
          y: rowIndex
        };
        if (item === 1) {
          if (loadTerm && this.animationId != 2) {
            objects.push(new Flag(this, position));
            this.terminal_cords.push(position);
          }

          // console.log(position);
        }
        if (item === 2) {
          objects.push(new House(this, position));
          this.blocking_cords.push(position);
        }
        if (item === 3) {
          if (loadTerm) {
            objects.push(new Person(this, position));
            this.terminal_cords.push(position);
          }
        }
        if (item === 4) {
          objects.push(new Coin(this, position));
        }
        if (item === 5) {
          objects.push(new Garbage(this, position));
        }
        if (item === 6) {
          let tile = new OWTile(this, position);
          objects.push(tile);
          this.ow_cords.push(position);
          this.ow_obs.push(tile);
        }
        if (item === 7) {
          let tile = new OWTile(this, position);
          objects.push(tile);
          objects.push(new Flag(this, position));
          this.terminal_cords.push(position);
          this.ow_cords.push(position);
          this.ow_obs.push(tile);
        }
        if (item === 8) {
          let tile = new OWTile(this, position);
          objects.push(tile);
          objects.push(new House(this, position));
          this.blocking_cords.push(position);
          this.ow_cords.push(position);
          this.ow_obs.push(tile);
        }
        if (item === 9) {
          let tile = new OWTile(this, position);
          objects.push(tile);
          if (loadTerm) {
            objects.push(new Person(this, position));
            this.terminal_cords.push(position);
            this.ow_obs.push(tile);
          }
          this.ow_cords.push(position);
        }
        if (item === 10) {
          let tile = new OWTile(this, position);
          objects.push(tile);
          objects.push(new Coin(this, position));
          this.ow_cords.push(position);
          this.ow_obs.push(tile);
        }
        if (item === 11) {
          let tile = new OWTile(this, position);
          objects.push(tile);
          objects.push(new Garbage(this, position));
          this.ow_cords.push(position);
          this.ow_obs.push(tile);
        }
      });
    });
    this.board_objects = objects;
  }
}
