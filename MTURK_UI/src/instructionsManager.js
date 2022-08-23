import Game from "/src/game";

export default class InstructionManager {
  constructor() {
    window.alpha =1
    window.delta = 0.015;
    this.finishedIns = false;
    this.insScene = 1;
    this.finishedGamePlay = false;
    this.animationGame1 = new Game(
      600,
      600,
      { x: 0, y: 3 },
      "ins_4_board",
      false
    );
    this.animationGame1.start("",false,true,true,1);

    this.animationGame2 = new Game(
      600,
      600,
      { x: 4, y: 5 },
      "ins_8_board",
      false
    );
    this.animationGame2.start("",false,true,true,2);
    this.showMidGameIns = false;
    this.midGameIns = 1;
    this.finishedMidGameIns = false;

  }

  // createButton(canvas, ctx) {}

  update() {

    if (this.insScene === 11 && !this.finishedGamePlay) {
      this.finishedIns = true;
    }
    if (this.insScene === 12 && this.finishedGamePlay && !window.finishedTrajBoard) {
      this.finishedIns = true;
      window.playTrajBoard = true;
    }
    // console.log(this.insScene)
    if (this.insScene === 15) {
      this.finishedIns = true;
      window.begunQueries = true;
    }

    //skips instruction where user explores gated area
    if (this.insScene == 9) this.insScene +=1;

    this.img = document.getElementById("ins" + String(this.insScene));
    if (this.showMidGameIns && this.midGameIns < 6){

      this.img = document.getElementById("mid_game_ins_" + String(this.midGameIns) + "_img");
      this.img_text = document.getElementById("mid_game_ins_" + String(this.midGameIns) + "_text");

    }
    else if (this.midGameIns >= 6){
      this.finishedMidGameIns = true;
      this.showMidGameIns = false;
    }
    if (this.insScene == 2 && (!window.isConnected || !window.retrievedSamples)) {
      console.log("failed to connect...")
      window.abort = true;
      this.img  = document.getElementById("error_message");
      //means there was an error connecting to the server
    }
	// this.img.setAttribute('width',  66);
	// this.img.setAttribute('height',  66);

    // $("#ins" + String(this.insScene)).show();
    // console.log("here");
  }
  playAnimation(ctx,id) {
    if (id === 1) {
      this.animationGame1.update();
      this.animationGame1.draw(ctx);
      // if (this.animationGame1.reached_terminal && ctx.globalAlpha > 0.5)ctx.globalAlpha-=0.001
    } else if (id === 2) {
      this.animationGame2.update();
      this.animationGame2.draw(ctx);
      // if (this.animationGame2.reached_terminal && ctx.globalAlpha > 0.5)ctx.globalAlpha-=0.001
      //ctx.globalAlpha = 1;
    }

  }
  draw(ctx) {
    this.pa1 = false;
    this.pa2 = false;

    if (this.insScene === 11 && !this.finishedGamePlay && !this.showMidGameIns) return;
    if (this.insScene === 15) return;
    if (this.insScene === 5) {
      this.pa2 = true;
    }
    if (this.insScene === 6) {
      this.pa1 = true;

      // this.h = 0.333*window.gsWidth;
      // this.w = 1.166*window.gsWidth;
      // this.x = -0.0666*window.gsWidth;
      // this.y = -0.0166*window.gsWidth;
      // this.h = 200;
      // this.w = 700;
      // this.x = -40;
      // this.y = -10;
    } else {
      //TODO: WE ONLY WANT TO DO THIS ON LOAD
      // if (!window.resizing) {
      //   this.w = 1.252*window.gsWidth;
      //   this.h = 0.751*window.gsWidth;
      //   this.x = -0.125*window.gsWidth;
      //   this.y = 0.083*window.gsWidth;
      //
      //
      // }

      //
      this.w = 750;
      this.h = 450;
      this.x = -75;
      this.y = 50;
    }

    let ins5_offset = 0;
    if (this.insScene == 5) {
      ins5_offset = -50;
    }


    if (this.showMidGameIns && this.midGameIns < 6){

      ctx.drawImage(this.img,250, 10, 345, 523);
      if (this.midGameIns == 1)ctx.drawImage(this.img_text, 60, 130, 127*1.4, 90*1.4);
      else if (this.midGameIns == 2)ctx.drawImage(this.img_text, 30, 70, 144.5*1.4, 209*1.4);
      else if (this.midGameIns == 3)ctx.drawImage(this.img_text, 60, 250, 120*1.4, 204.5*1.4);
      else if (this.midGameIns == 4)ctx.drawImage(this.img_text, 20, 270, 142.5*1.4, 234*1.4);
      else if (this.midGameIns == 5)ctx.drawImage(this.img_text, 15, 210, 174.5*1.4, 264.5*1.4);
        // ctx.drawImage(this.img,250, 10, 345, 535);

    } else {
      ctx.drawImage(this.img, this.x, this.y + ins5_offset, this.w, this.h);
      if (this.insScene == 1) {
        let label = document.getElementById("click_here_btn");
        ctx.drawImage(label, 445, 500, 150, 50);

      }
    }


    // $("#ins" + String(this.insScene)).show();
    if (this.pa1)this.playAnimation(ctx,1);
    if (this.pa2)this.playAnimation(ctx,2);
    //draw button
    // ctx.fillStyle = "white";
    // ctx.beginPath();
    // ctx.rect(500, 550, 100, 50);

    // ctx.font = String(0.04166*window.gsWidth) + "px CustomFont";
    ctx.font = "20px CustomFont";

    ctx.fillStyle = "gray";

    if (!window.abort) {
      if (this.insScene != 5 && this.insScene != 6) {
        if (this.showMidGameIns && this.midGameIns <=5)ctx.fillText("Next",500, 570);
        else if (this.insScene === 10) ctx.fillText("Play", 500, 570);
        else if (this.insScene === 11) ctx.fillText("Play", 500, 570);
        else if (this.insScene === 14) ctx.fillText("Begin", 500, 570);
        else ctx.fillText("Next",500, 570);
      } else {
        if (this.insScene == 5 && this.animationGame2.finishedAnimation) {
          ctx.fillText("Next",500, 570);
        } else if (this.insScene == 6 && this.animationGame1.finishedAnimation) {
          ctx.fillText("Next",500, 570);
        }
      }
      ctx.fill();
    }
    // console.log("here");


    //500, 570

  }
}
