import 'regenerator-runtime/runtime'
// import regeneratorRuntime from "regenerator-runtime";

import Game from "/src/game";
import Score from "/src/score";
import InstructionsManager from "/src/instructionsManager";
import QueryManager from "/src/queryManager"
//---------------------------------MTURK STUFF--------------------------------------
// var AWS = require('aws-sdk');
// var region_name = 'us-east-1';
// var endpoint = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com';
// // Uncomment this line to use in production
// //var endpoint = 'https://mturk-requester.us-east-1.amazonaws.com';
// AWS.config.update({ region: 'us-east-1',
// 		    endpoint: endpoint });
//
// var mturk = new AWS.MTurk();
// //
// let QUESTION_XML = '<ExternalQuestion' +
//  'xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">' +
//  '<ExternalURL>"https://dev.d1e16e31dm86sr.amplifyapp.com"</ExternalURL>' +
//  '<FrameHeight>500</FrameHeight>' + '</ExternalQuestion>'
//

//---------------------------------MTURK STUFF--------------------------------------

window.canvas = document.getElementById("gameScreen");
let ctx = window.canvas.getContext("2d");

let scoreDisp = document.getElementById("scoreScreen");
let ctxScore = scoreDisp.getContext("2d");

let queryDisp = document.getElementById("queryScreen");
let ctxQuery = queryDisp.getContext("2d");

// var svg = document.querySelector("svg");

const GAME_WIDTH = 600;
const GAME_HEIGHT = 600;
const QUERY_WIDTH = 1200;
const QUERY_HEIGHT = 700;
const SCORE_WIDTH = 300;
const SCORE_HEIGHT = 600;
//width="300" height="600"
const CELL_SIZE = GAME_WIDTH / 10;
//
//
// console.log(window.innerWidth);
// console.log(window.innerHeight);
//
// window.canvas.width = GAME_WIDTH;
// window.canvas.height = GAME_HEIGHT;

//
// window.canvas.width = 0.416*window.innerWidth
// window.canvas.height = 0.416*window.innerWidth

// console.log((window).width());
// console.log(window.innerWidth);
window.canvas.width = $(window).height()
window.canvas.height = $(window).height()

window.gsWidth = window.canvas.width;
window.gsHeight = window.canvas.height;

window.qdWidth = 0.9*$(window).width();
window.qdHeight=  window.qdWidth/2;

window.canvas.style.left = "18%";
window.canvas.style.top = "0%"
window.canvas.style.position = "absolute";
//
// scoreDisp.width = 300;
// scoreDisp.height = 600;
scoreDisp.width = 0.208*$(window).width();
scoreDisp.height = $(window).height()



scoreDisp.style.left ="68%"
// scoreDist.style.left =  String((window.canvas.width/$(window).width()) + 10 + 10) + "%";
// console.log(String(10*(window.canvas.width/$(window).width()) + 10 + 10) + "%")
scoreDisp.style.top = "0%";
scoreDisp.style.position = "absolute";

// queryDisp.width = QUERY_WIDTH;
// queryDisp.height = QUERY_HEIGHT;
// queryDisp.width = 0.9*$(window).width();
// queryDisp.height =  1*$(window).height()

queryDisp.style.left = "5%";
// $('#queryScreen').css('width', '90%');
$('#queryScreen').css('height', '100%');


queryDisp.style.top = "0%";
queryDisp.style.position = "absolute";

window.addEventListener("keydown", function(e) {
    if(["Space","ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].indexOf(e.code) > -1) {
        e.preventDefault();
    }
}, false);

window.addEventListener("load", function(event) {
  // window.im.w = 1.252*window.gsWidth;
  // window.im.h = 0.751*window.gsWidth;
  // window.im.x = -0.125*window.gsWidth;
  // window.im.y = 0.083*window.gsWidth;
  // window.im.nW =0.833*window.gsWidth;
  // window.im.nH =0.95*window.gsWidth;
  //
  ctx.canvas.width  = 600;
  ctx.canvas.height = 600;

  ctxQuery.canvas.width = 1009;
  ctxQuery.canvas.height = 699;

  scoreDisp.width = 300;
  scoreDisp.height = 600;

  var gsRect = window.canvas.getBoundingClientRect();
  let gsLX = gsRect.left;
  var ssRect = scoreDisp.getBoundingClientRect();
  let ssLX = ssRect.left;
  let maxDist = 10;
  let newGsSize = $(window).height();
  let newCsWidth = 0.208*$(window).width();
  let newCsHeight = $(window).height();
  let newCsDispWidth = 0.208*$(window).width();
  let newCsDispHeight =$(window).height();

  if (window.playTrajBoard || (window.n_games < window.max_games)) {
    // if (gsLX + $(window).height() - ssLX >= maxDist) {
    //   //means we are overlapping
    //   newGsSize = $(window).height() - (gsLX + $(window).height() - ssLX);
    //   // newCsWidth =8*newCsWidth/10
    // }
    if ($(window).height() < 500 || gsLX + $(window).height() - ssLX >= maxDist ) {
      // newCsDispWidth = 0.4*$(window).width();
      // newCsDispHeight = 1.9*$(window).height();
      newCsDispWidth = 0.3*$(window).width();
      newCsDispHeight = 1.7*$(window).height();
    }

  }

  //WE NEED TO RESIZE GAME SCREEN BASED ON THE DISTANCE FROM GAME SCREEN TO SCORE

  var gs = $('#gameScreen');
  gs.css("width", newGsSize);
  gs.css("height", newGsSize);

  var sd = $('#scoreScreen');
  sd.css("width", newCsWidth);
  sd.css("height",newCsHeight);
  sd.css("left",gsLX + newGsSize + 10);

  var qs = $('#queryScreen');
  // qs.css("width", 0.9*$(window).width());
  // qs.css("height", newGsSize);
  // qs.css("height", newGsSize);
  $('#queryScreen').width($('#queryScreen').height()*2);
  if ($('#queryScreen').height()*2 > 0.9*$(window).width()){
    $('#queryScreen').width(0.9*$(window).width());
  }

  var bs = $('.displayInsBtn');
  bs.css("width", 0.1*$(window).width());
  bs.css("height", 0.075*$(window).height());
  bs.css("font-size", 0.02*$(window).height());
  bs.css("left", gsLX + newGsSize/2 - (0.1*$(window).width())/2);

  window.gsWidth = newGsSize;
  window.gsHeight = newGsSize;

  window.qdHeight=  $(window).height();
  window.qdWidth = 2*window.qdHeight;

  if (2*window.qdHeight > 0.9*$(window).width()) {
    window.qdWidth = 0.9*$(window).width()
  }

  scoreDisp.width = newCsDispWidth;
  scoreDisp.height = newCsDispHeight;

  //
  rects = updateRects(window.gsWidth, window.gsWidth);


  window.nextRect = rects[0];
  window.leftRect = rects[1];
  window.rightRect = rects[2];
  window.sameRect = rects[3];
  window.incRect = rects[4];
  window.submitSurveyRect = rects[5];
  window.openLinkRect = rects[6]


  // console.log(window.gsWidth);

});

// function resizeend() {
//     if (new Date() - rtime < delta) {
//         setTimeout(resizeend, delta);
//     } else {
//         timeout = false;
//         window.resizing = false;
//     }
// }

// var rtime;
// var timeout = false;
// var delta = 200;
window.addEventListener("resize", function(event) {
  window.resizing = true;
  // if (timeout === false) {
  //       timeout = true;
  //       setTimeout(resizeend, delta);
  //   }
  // let dw = window.innerWidth/window.canvas.width;
  // let hw = window.innerHeight/window.canvas.height;
  // console.log(dw);
  // console.log(hw);
  // ctx.scale(dw, hw);
  var gsRect = window.canvas.getBoundingClientRect();
  let gsLX = gsRect.left;
  var ssRect = scoreDisp.getBoundingClientRect();
  let ssLX = ssRect.left;
  let maxDist = 10;
  let newGsSize = $(window).height();

  let newCsWidth = 0.208*$(window).width();
  let newCsHeight = $(window).height();

  let newCsDispWidth = 0.208*$(window).width();
  let newCsDispHeight =$(window).height();
  // let newCsDispWidth = 0.6*$(window).width();
  // let newCsDispHeight = 1.2*$(window).height();
  if (window.playTrajBoard || (window.n_games < window.max_games)) {
    // if (gsLX + $(window).height() - ssLX >= maxDist) {
    //   //means we are overlapping
    //   newGsSize = $(window).height() - (gsLX + $(window).height() - ssLX);
    //   // newCsWidth =8*newCsWidth/10
    //
    // }
    if ($(window).height() < 500 || gsLX + $(window).height() - ssLX >= maxDist ) {

      // newCsDispWidth = 0.4*$(window).width();
      // newCsDispHeight = 1.9*$(window).height();

      newCsDispWidth = 0.3*$(window).width();
      newCsDispHeight = 1.7*$(window).height();

    }

    // if (newCsWidth > (0.42)*newGsSize) {
    //   newCsDispWidth = 0.5*$(window).width();
    //   newCsDispHeight = 2*$(window).height();
    // }

  }


  var gs = $('#gameScreen');
  gs.css("width", newGsSize);
  gs.css("height", newGsSize);

  var sd = $('#scoreScreen');
  sd.css("width", newCsWidth);
  sd.css("height",newCsHeight);
  sd.css("left",gsLX + newGsSize + 10);

  var qs = $('#queryScreen');
  // qs.css("width", 0.9*$(window).width());
  // qs.css("height", newGsSize);
  $('#queryScreen').width($('#queryScreen').height()*2);
  if ($('#queryScreen').height()*2 > 0.9*$(window).width()){
    $('#queryScreen').width(0.9*$(window).width());
  }

  var bs = $('.displayInsBtn');
  bs.css("width", 0.1*$(window).width());
  bs.css("height", 0.075*$(window).height());
  bs.css("font-size", 0.02*$(window).height());
  bs.css("left", gsLX + newGsSize/2 - (0.1*$(window).width())/2);


  // window.gsWidth = $(window).height();
  // window.gsHeight = $(window).height();
  window.gsWidth = newGsSize;
  window.gsHeight = newGsSize;

  window.qdHeight=  $(window).height();
  window.qdWidth = 2*window.qdHeight;

  if (2*window.qdHeight > 0.9*$(window).width()) {
    window.qdWidth = 0.9*$(window).width()
  }

  scoreDisp.width = newCsDispWidth;
  scoreDisp.height = newCsDispHeight;

  // scoreDisp.width = 0.208*$(window).width();
  // scoreDisp.height = $(window).height()

  rects = updateRects(window.gsWidth, window.gsWidth);


  window.nextRect = rects[0];
  window.leftRect = rects[1];
  window.rightRect = rects[2];
  window.sameRect = rects[3];
  window.incRect = rects[4];
  window.submitSurveyRect = rects[5]
  window.openLinkRect = rects[6]
  //
  // window.im.w = 750;
  // window.im.h = 450;
  // window.im.x = -75;
  // window.im.y = 50;


});

//set SVG (for curves as an element of the canvas)

window.spawnPoints = [];
window.boardNames = [];
window.instructions = [];

// const spawnPoint1 = { x: 0, y: 5 };
// window.spawnPoints.push(spawnPoint1);
// const boardName1 = "player_board_1";
// window.boardNames.push(boardName1);

const spawnPoint2 = { x: 0, y: 5 };
window.spawnPoints.push(spawnPoint2);
window.spawnPoints.push(spawnPoint2);

const boardName2 = "player_board_1";
window.boardNames.push(boardName2);
window.boardNames.push(boardName2);

window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_1_ins.png");
window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_1_ins.png");

const spawnPoint3 = { x: 0, y: 8 };
window.spawnPoints.push(spawnPoint3);
window.spawnPoints.push(spawnPoint3);

const boardName3 = "redone_player_board_2";
window.boardNames.push(boardName3);
window.boardNames.push(boardName3);

window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_2_ins.png");
window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_2_ins.png");


const spawnPoint4 = { x: 8, y: 9 };
window.spawnPoints.push(spawnPoint4);
window.spawnPoints.push(spawnPoint4);

const boardName4 = "redone_player_board_3";
window.boardNames.push(boardName4);
window.boardNames.push(boardName4);

window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_3_ins.png");
window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_3_ins.png");

const spawnPoint5 = { x: 9, y: 9 };
window.spawnPoints.push(spawnPoint5);
window.spawnPoints.push(spawnPoint5);

const boardName5 = "redone_player_board_4";
window.boardNames.push(boardName5);
window.boardNames.push(boardName5);

window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_4_ins.png");
window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_4_ins.png");


const spawnPoint6 = { x: 8, y: 9 };
window.spawnPoints.push(spawnPoint6);
window.spawnPoints.push(spawnPoint6);

const boardName6 = "player_board_5";
window.boardNames.push(boardName6);
window.boardNames.push(boardName6);

window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_5_ins.png");
window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_5_ins.png");

const spawnPoint7 = { x: 5, y: 5 };
window.spawnPoints.push(spawnPoint7);
window.spawnPoints.push(spawnPoint7);

const boardName7 = "player_board_6";
window.boardNames.push(boardName7);
window.boardNames.push(boardName7);

window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_6_ins.png");
window.instructions.push("https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/board_6_ins.png");

//NOTE: USE THIS LINK TO VIEW TRAJECTORIES https://codesandbox.io/s/gridworld-dwkdg?file=/src/trajHandler.js
window.disTraj = false;
window.im = new InstructionsManager();
window.qm = new QueryManager();
window.isConnected = false;
//this.socket.send(this.cur_name);
// ws://localhost:3000
window.socket = new WebSocket("wss://f5c764accdb1.ngrok.io");
    window.socket.addEventListener("open", function (event) {
      window.isConnected = true;
      console.log("Connected to the WS Server!");

    });



function waitForCondition() {
  return new Promise(resolve => {
    var start_time = Date.now();
    function checkFlag() {
      if (window.isConnected) {
        resolve();
      }else {
        window.setTimeout(checkFlag, 1000);
      }

    }
    checkFlag();
  });
}

async function getInstanceParams() {
  //make sure the worker has accepted our HIT
  if (window.qm.assignmentID != "ASSIGNMENT_ID_NOT_AVAILABLE") {
      //wait for the server to connect and then send our message
      await waitForCondition();
      console.log("sent");
      window.socket.send(window.qm.assignmentID);
  }
}

// Listen for messages
window.socket.addEventListener("message", function (event) {
  window.sampleNumber = parseInt(event.data.split(",")[0])
  window.observationType = parseInt(event.data.split(",")[1])
  if (window.sampleNumber >=0 && window.sampleNumber <= window.nSamples && window.observationType >= 0 && window.observationType <=2) {
    window.qm.handleSampleImages()
    window.qm.nTrajs = window.nSamples;
    window.retrievedSamples = true;
  }

});

//------------------------------------------------
let nTrajBoards = 0;
window.showFinalScreen = false;
window.game = null;
window.playTrajBoard = false;
window.finishedTrajBoard = false;
window.n_games = 0;
window.max_games = 11;
window.finished_game = true;
window.total_tsteps = 50;
window.timestep = 0;
window.finishedHIT = false;
// window.sampleNumber = getRandomInt(0,2);
window.nSamples = 37;
window.nUsers = 30;
window.abort = false;
// window.sampleNumber = getRandomInt(0,window.nUsers);; //TODO: CHANGE ONCE WE MAKE MORE SAMPLE SETS
getInstanceParams();
// window.observationType = getRandomInt(0,2); //0 means display partial return and 1 means display change in state value
// window.observationType = 0; //TODO: CHANGE ONCE WE MAKE MORE SAMPLE SETS

window.openedSurvey = false;

// window.im.createButton(canvas, ctx);

//creates buttons
window.rects = updateRects(window.canvas.width, window.canvas.height);
window.nextRect = rects[0];
window.leftRect = rects[1];
window.rightRect = rects[2];
window.sameRect = rects[3];
window.incRect = rects[4];
window.submitSurveyRect = rects[5]
window.openLinkRect = rects[6]
//------------------------------------------------
//https://stackoverflow.com/questions/24384368/simple-button-in-html5-canvas/24384882
function getMousePos(canvas, event) {
  var rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

function isInside(pos, rect) {
  return (
    pos.x > rect.x &&
    pos.x < rect.x + rect.width &&
    pos.y < rect.y + rect.height &&
    pos.y > rect.y
  );
}


function updateRects(w,h) {

  var nextRect = {
    x: 0.833*w,
    y: 0.916*w,
    width: 100,
    height: 50
  };

  // this.offset = 500
  // this.b1_x = 200+(420/2)-(70/2)-100-125
  // this.b1_y = 500-100+15
  // this.b1_h = 50
  // this.b1_w = 70
  // this.b2_x = 290+(420/2)-(70/2)-100+this.offset-125
  // this.b3_x = 720-160/2-100-125
  // this.b3_y = 425-100+15
  // this.b3_h = 130
  // this.b3_w = 50
  // this.b4_y = 500-100+15
  var submitSurveyRect = {
    x: (window.qm.sb_x/1009)*window.qdWidth,
    y: (window.qm.sb_y/699)*window.qdHeight,
    width: ((window.qm.sb_w*1.5)/1009)*window.qdWidth,
    height: ((window.qm.sb_h*1.5)/699)*window.qdHeight

  }

  var openLinkRect = {
    x: (window.qm.sb_x/1009)*window.qdWidth,
    y: ((window.qm.sb_y-50)/699)*window.qdHeight,
    width: ((window.qm.sb_w*1.5)/1009)*window.qdWidth,
    height: ((window.qm.sb_h*1.5)/699)*window.qdHeight

  }

  var leftRect = {
    x: (window.qm.b1_x/1009)*window.qdWidth,
    y: (window.qm.b1_y/699)*window.qdHeight,
    width: ((50*1.5)/1009)*window.qdWidth,
    height: ((70*1.5)/699)*window.qdHeight
  };

  var rightRect = {
    x: (window.qm.b2_x/1009)*window.qdWidth,
    y: (window.qm.b1_y/699)*window.qdHeight,
    width: ((50*1.5)/1009)*window.qdWidth,
    height: ((70*1.5)/699)*window.qdHeight
  };


  var sameRect = {
    x:(window.qm.b3_x/1009)*window.qdWidth,
    y: (window.qm.b3_y/699)*window.qdHeight,
    width: ((130*1.5)/1009)*window.qdWidth,
    height: ((50*1.5)/699)*window.qdHeight
  };

  var incRect = {
    x: (window.qm.b3_x/1009)*window.qdWidth,
    y: (window.qm.b4_y/699)*window.qdHeight,
    width: ((130*1.5)/1009)*window.qdWidth,
    height: ((50*1.5)/699)*window.qdHeight
  };
  return [nextRect, leftRect, rightRect, sameRect, incRect,submitSurveyRect,openLinkRect]
}


//Binding the click event on the canvas
window.canvas.addEventListener(
  "click",
  function (evt) {
    let mousePos = getMousePos(window.canvas, evt);

    if (isInside(mousePos, window.nextRect) && !this.finishedIns && !(window.im.insScene == 5 && !window.im.animationGame2.finishedAnimation) && !(window.im.insScene == 6 && !window.im.animationGame1.finishedAnimation) && !window.abort) {
      if (!window.im.showMidGameIns)window.im.insScene += 1;
      if (window.im.showMidGameIns)window.im.midGameIns+=1;
      // window.alpha = 1;
      ctx.globalAlpha = 1;
      window.alpha = 1;
    }
  },
  false
);

queryDisp.addEventListener(
  "click",
  function (evt) {
    let mousePos = getMousePos(queryDisp, evt);
    if (isInside(mousePos,window.submitSurveyRect) && window.showFinalScreen && window.openedSurvey) {
      window.finishedHIT = true;
      // window.open("https://docs.google.com/forms/d/e/1FAIpQLScx9ngKDyBEWVwUH1bTey1Km7mt1FH3tjxvfACFCd4ERZ5A6Q/viewform?usp=sf_link");
    } else if (isInside(mousePos,window.openLinkRect) && window.showFinalScreen) {
      window.open("https://docs.google.com/forms/d/e/1FAIpQLScx9ngKDyBEWVwUH1bTey1Km7mt1FH3tjxvfACFCd4ERZ5A6Q/viewform?usp=sf_link");
      window.openedSurvey = true;
    }
    else if (isInside(mousePos,window.leftRect) && window.begunQueries) {
      window.qm.pressed = true;
      window.qm.queried("left");

    } else if (isInside(mousePos,window.rightRect) && window.begunQueries) {
      window.qm.pressed = true;
      window.qm.queried("right");
    }else if (isInside(mousePos,window.sameRect) && window.begunQueries) {
      window.qm.pressed = true;
      window.qm.queried("same");
    }else if (isInside(mousePos,window.incRect) && window.begunQueries) {
      window.qm.pressed = true;
      window.qm.queried("dis");
    }
  },
  false
);
function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
}
//------------------------------------------------
function checkInsCompletion() {
  //check objective and see if we still need to play another board
  if (window.n_games === 1){
    if (window.game.hitFlag)window.n_games += 1;
  }else if (window.n_games === 3){
      window.n_games += 1;
  }else if (window.n_games === 5){
    if (window.game.hitFlag)window.n_games += 1;
  }else if (window.n_games === 7){
    if (window.game.hitFlag)window.n_games += 1;
  }else if (window.n_games === 9){
    if (window.game.hitEdge && window.game.hitHouse)window.n_games += 1;
  }
  // else if (window.n_games === 11){
  //   if (window.game.hitPerson)window.n_games += 1;
  // }

}

function startNewGame() {
  if (window.finished_game && ((window.n_games < window.max_games) || window.playTrajBoard) ) {
    console.log("starting new game...");
    console.log(window.n_games);
    window.finished_game = false;
    window.timestep = 0;
    let boardName;
    let spawnPoint;
    let ins = "";
    let dispIns = true;
    if (window.disTraj) {
      boardName = "2021-07-29_sparseboard2-notrap";
      spawnPoint = { x: 0, y: 0 };
    } else if (window.playTrajBoard) {
      boardName = "2021-07-29_sparseboard2-notrap";
      if (nTrajBoards === 0)spawnPoint ={ x: 0, y: 0 };
      else spawnPoint ={ x: 0, y: 5 };
      dispIns = false;
    } else {
      boardName = window.boardNames[window.n_games];
      spawnPoint = window.spawnPoints[window.n_games];
      ins = window.instructions[window.n_games];
    }

    if (window.n_games == 4 && !window.im.finishedMidGameIns) {
      window.im.showMidGameIns = true;

    }

    // if (boardName==="player_board_6") {
    //   console.log("here")
    //   window.n_games = window.max_games
    //   window.game.reached_terminal = true;
    //   window.im.finishedIns = false;
    //   window.im.finishedGamePlay = true;
    //   window.finished_game = true;
    //   return
    //   //play the trajectory board
    // }
    // this.finishedAnimation = false;
    window.game = new Game(
      GAME_WIDTH,
      GAME_HEIGHT,
      spawnPoint,
      boardName,
      window.disTraj
    );
    window.game.start(ins=ins,dispIns=dispIns);
    window.score = new Score(window.game);
    window.score.start();

    window.lastTime = 0;
    window.alpha = 1; /// current alpha value
    window.delta = 0.008; /// delta = speed

    window.n_games += 1;

    window.begunQueries = false;
    ctx.globalAlpha = 1;

    // if (window.n_games === window.max_games) {
    //   return;
    // }
  }
}

function gameLoop(timestamp) {
  ctx.clearRect(0, 0, window.canvas.width, window.canvas.height);
  ctxScore.clearRect(0, 0, scoreDisp.width, scoreDisp.height);
  ctxQuery.clearRect(0,0,queryDisp.width,queryDisp.height);
  let deltaTime = timestamp - window.lastTime;

  if (!window.im.finishedIns && !window.disTraj) {
    // console.log("here")
    // ctx.globalAlpha = 1;
    window.im.update();
    window.im.draw(ctx);
    // console.log(window.im.animationGame1.finishedAnimation)
    if (((window.im.animationGame2.finishedAnimation && window.im.insScene == 5)||(window.im.animationGame1.finishedAnimation && window.im.insScene == 6))) {
      //figure out how to fade
      // window.alpha -= window.delta;
      // ctx.globalAlpha = window.alpha;
      ctx.beginPath();
      ctx.fillStyle = "white";
      ctx.globalCompositeOperation='soft-light';
      ctx.fillRect(0, 190, 600, 350);
      ctx.fillRect(0, 540, 400, 350);
      ctx.stroke();
    }

    requestAnimationFrame(gameLoop);
    // requestAnimationFrame(gameLoop);
  } else if (window.begunQueries) {
    ctx.globalAlpha = 1;
    window.qm.update(deltaTime);
    window.qm.draw(ctxQuery);

    requestAnimationFrame(gameLoop);

  }else {
    startNewGame();
    window.lastTime = timestamp;

    if ((window.game.reached_terminal === true && !window.disTraj) || window.finishedAnimation) {

      //figure out how to fade
      window.alpha -= window.delta;
      // if (alpha <= 0 || alpha >= 1) delta = -delta;
      ctx.globalAlpha = window.alpha;
      //clear a bunch of stuff like incompleteCords here
      //if user was playing game
    }

    if (window.alpha > 0) {
      if (window.game.gameObjects != null && !window.im.showMidGameIns) {
        window.game.update(deltaTime);
        window.game.draw(ctx);
        window.score.update(deltaTime);
        window.score.draw(ctxScore);

        if (!window.playTrajBoard && (window.n_games >= window.max_games)) {
          window.game.reached_terminal = true;
          window.im.finishedIns = false;
          window.im.finishedGamePlay = true;
          window.finished_game = true;
          $(".displayInsBtn").hide();
        }
      } else if (window.im.showMidGameIns && !window.im.finishedMidGameIns){
          ctx.globalAlpha = 1;
          window.im.update();
          window.im.draw(ctx);
      }
    } else {
      window.finished_game = true;
      checkInsCompletion();
      if (window.playTrajBoard && nTrajBoards === 0){
        nTrajBoards+=1;
        // window.n_games-=1;
      }else if (window.playTrajBoard && nTrajBoards === 1){
        window.playTrajBoard = false;
        window.im.finishedIns = false;
        window.im.finishedGamePlay = true;
        window.finishedTrajBoard = true;
        ctx.globalAlpha = 1;
        // window.im.insScene+=1;
      }
      // window.playTrajBoard = false;
    }
    requestAnimationFrame(gameLoop);
  }
}

requestAnimationFrame(gameLoop);
