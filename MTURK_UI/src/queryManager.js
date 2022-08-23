import QueryInputHandler from "/src/queryInput";

export default class InstructionManager {

  constructor() {
    this.n_query = 0;
    // this.img = document.getElementById("img_ow_tile");

    // this.nTrajs = 1;
    this.traj = 0;
    this.started = false;
    this.pressed = false;
    // ASSIGNS THE URL PARAMETERS TO JAVASCRIPT VARIABLES
    this.assignmentID = this.turkGetParam('assignmentId');
    this.b1Color = "grey";
    this.b2Color = "grey";
    this.b3Color = "grey";
    this.b4Color = "grey";
    this.isSubmitted = false;
    this.addedLink = false;

      //window.sampleNumber

    // this.offset = 0.416*window.qdWidth;
    // this.b1_x = 0.229*window.qdWidth;
    // this.b1_y = 0.714*window.qdHeight;
    // this.b1_h = 50
    // this.b1_w = 70
    // this.b2_x = (0.304*window.qdWidth)+this.offset
    // this.b3_x = 0.45*window.qdWidth
    // this.b3_y = 0.607*window.qdHeight;
    // this.b3_h = 130
    // this.b3_w = 50
    // this.b4_y = 0.714*window.qdHeight;
    this.offset = 500
    this.b1_x = 200+(420/2)-(70/2)-100-125 - 10 + 23
    this.b1_y = 500-100+15 + 40 + 70 + 130 - 30
    this.b1_h = 50
    this.b1_w = 70
    this.b2_x = 290+(420/2)-(70/2)-100+500-125 - 10+ 23
    this.b3_x = 720-160/2-100-125 - 10 + 23
    this.b3_y = 425-100+15 + 40 + 70 + 130 - 30
    this.b3_h = 130
    this.b3_w = 50
    this.b4_y = 500-100+15 + 40 + 70 + 130 - 30

    this.sb_x = 200+(420/2)-(70/2)
    this.sb_x_d = 200+(420/2)-(70/2)-20
    this.sb_y = 500-100+15
    this.sb_w = 175
    this.sb_h = 100


    //load in images from the folder corrospding to window.sampleNumber


    new QueryInputHandler();
  }

  handleSampleImages(){
    let gitDir;

    if (window.observationType == 0){
      gitDir = "https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/2021_07_29_data_samples/pr_sample" + String(window.sampleNumber) + "/"
    } else if (window.observationType == 1) {
      gitDir = "https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/2021_07_29_data_samples/vf_sample" + String(window.sampleNumber) + "/"
    }else if (window.observationType == 2) {
      gitDir = "https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/2021_07_29_data_samples/none_sample" + String(window.sampleNumber) + "/"
    }
    // else if (window.observationType == 3) {
    //   gitDir = "https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/2021_07_29_data_samples/none_sample" + String(window.sampleNumber) + "/"
    // }
    for (let i = 0; i < window.nSamples; i++) {
      //create MTURK HIT query
      // <input type="hidden" id = "query3" name="query3" value="" /> -->
      var q = document.createElement('input');
      q.setAttribute('type', 'text');
      q.type = "hidden"
      q.id = "query" + String(i)
      q.name = "query" + String(i)
      q.value = ""

      document.getElementById('hitForm').appendChild(q);


      //create/store image
      let sample1 = gitDir + "s" + String(i) + "_0" + ".png"
      let sample2 = gitDir + "s" + String(i) + "_1" + ".png"
      //traj10
      var img1 = document.createElement('img');
      img1.src =sample1;
      img1.id = "traj" + String(i) + "0"
      document.getElementById('body').appendChild(img1);
      //traj11
      var img2 = document.createElement('img');
      img2.src =sample2;
      img2.id = "traj" + String(i) + "1"
      document.getElementById('body').appendChild(img2);
    }
  }

  turkGetParam( name ) {
    let fullurl = window.location.href;
    var regexS = "[\?&]"+name+"=([^&#]*)";
    var regex = new RegExp( regexS );
    var tmpURL = fullurl;
    var results = regex.exec( tmpURL );
    if( results == null ) {
        return "";
    } else {
        return results[1];
    }
}

  start(deltaTime) {
    // this.offset = 0.495*window.qdWidth;
    // this.b1_x = 0.272*window.qdWidth;
    // this.b1_y = 1.02*window.qdHeight;
    // this.b1_h = 50
    // this.b1_w = 70
    // this.b2_x = 0.361*window.qdWidth+this.offset
    // this.b3_x = 0.535*window.qdWidth
    // this.b3_y = 0.869*window.qdHeight;
    // this.b3_h = 130
    // this.b3_w = 50
    // this.b4_y = 1.022*window.qdHeight;

    // window.canvas.width = window.innerWidth;
    // window.canvas.height = window.innerHeight;
    // window.canvas.width = 1200;
    // window.canvas.height = 700;
    window.canvas.style.display = "none"
    this.started = true;
    this.queryResults = {}
    this.lastDeltaTime = deltaTime;

    // var gs = $('#gameScreen');
    // gs.css("width", newGsSize);
    // gs.css("height", newGsSize);

    $("#myModal").on('show.bs.modal', function (e) {
       var modal = $(this)

       var mimg = $('.img-responsive');
       mimg.css("width", 700);
       mimg.css("height", 350);
       // modal.find('.label1').text(ins)
       // modal.find('.modal_img1').attr("src","assets/images/img_flag.png")
       $(".img-responsive").attr('src', "https://raw.githubusercontent.com/Stephanehk/Prefrence_Elicitation_Interface/main/assets/images/query_ins.png");
       // $(".img-responsive").attr('width', "500");
       // $(".img-responsive").attr('height', "500");

       //modal_img1

    });
    $("#myModal").modal("show");
  }

  submit() {
    // console.log(this.assignmentID);
    if (!this.isSubmitted) {
      document.getElementById('assignmentId').value = this.assignmentID
      document.getElementById('sampleNumber').value = window.sampleNumber
      document.getElementById('observationType').value = window.observationType

      // for (var key in this.queryResults) {
      //   document.getElementById('foo').value = this.queryResults[key]
      //
      // }
      // document.forms[0].submit()
      document.getElementById('hitForm').submit();
      this.isSubmitted = true;
    }


  }

  resetButtons() {
    this.b1Color = "grey";
    this.b2Color = "grey";
    this.b3Color = "grey";
    this.b4Color = "grey";

  }

  queried(data) {
    // this.queryResults["traj" + String(this.traj)] = data;
    if (data === "left")this.b1Color = "black";
    if (data === "right")this.b2Color = "black";
    if (data === "same")this.b3Color = "black";
    if (data === "dis")this.b4Color = "black";

    document.getElementById('query'+ String(this.traj)).value = data

  }

  update(deltaTime) {
    if (!this.started)this.start(deltaTime);
    //
    // this.offset = 0.416*window.qdWidth;
    // this.b1_x = 0.229*window.qdWidth;
    // this.b1_y = 0.714*window.qdHeight;
    // this.b1_h = 50
    // this.b1_w = 70
    // this.b2_x = (0.304*window.qdWidth)+this.offset
    // this.b3_x = 0.45*window.qdWidth
    // this.b3_y = 0.607*window.qdHeight;
    // this.b3_h = 130
    // this.b3_w = 50
    // this.b4_y = 0.714*window.qdHeight;
    if (window.finishedHIT){
      this.submit();
    }

    if (deltaTime -  this.lastDeltaTime > 200){
      this.resetButtons();
      this.lastDeltaTime = deltaTime;
    }
    if (this.pressed){
      // console.log("pressed");
      // console.log(this.traj);
      if (this.traj === this.nTrajs-1) {
        window.showFinalScreen = true;
        // this.b1_x = this.sb_x;
        // this.b1_y = this.sb_y;
        // this.b1_h = this.sb_h;
        // this.b1_w = this.sb_w;
        // this.submit();
      } else {
        this.traj+=1;
        this.pressed = false;
      }
    }
    // this.img.src = "/assets/trajImages/(0,2)_0_1.png"
    let trajId1 = "traj" + String(this.traj) + "0";
    let trajId2 = "traj" + String(this.traj) + "1";
    this.img1 = document.getElementById(trajId1);
    this.img2 = document.getElementById(trajId2);

  }

  drawButton(ctx, x, y, w, h, color,text,tx,ty) {
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
    ctx.stroke();

    ctx.font = "20px CustomFont";
    ctx.fillStyle = "white";
    ctx.fillText(text, tx,ty);
    ctx.fill();

  }

  // postRequest(data) {
  //
  //   router.post('https://workersandbox.mturk.com/mturk/externalSubmit',function(req,res,next) {
  //     var id = req.params.id;
  //     res.redirect("/test/...");
  //   });
  // }

  draw(ctx) {

    if (window.showFinalScreen) {
      this.img = document.getElementById("ins15");
      this.w = 750;
      this.h = 450;
      this.x = 75;
      this.y = 0;


      if (window.openedSurvey){
        this.drawButton(ctx,this.sb_x-5,this.sb_y,this.sb_w,this.sb_h,"DodgerBlue","Submit HIT",this.sb_x-(this.sb_w/2)+120-5,this.sb_y+(this.sb_h/2)+25);

      }

      // this.drawButton(ctx,this.b1_x,this.b1_y,this.b1_w,this.b1_h,this.b1Color,"Left",this.b1_x-(this.b1_w/2)+45,this.b1_y+(this.b1_h/2)+5);

      ctx.drawImage(this.img, this.x, this.y, this.w, this.h);

      ctx.font = "25px CustomFont";
      ctx.fillStyle = "black";
      ctx.fillText("Open Survey", this.sb_x,this.sb_y-30);
      ctx.fill();


    } else {
      ctx.font = "25px CustomFont";
      ctx.fillStyle = "black";
      ctx.fillText("Which Shows", 480-40-20-30+10-5-5+20+7,20+130);
      ctx.fillText("Better", 480-40-20-30+10-5+20+30+10,20+160);
      ctx.fillText("Behavior?", 480-40-20-30+10-5+20+15+7,20 + 190);
      ctx.fill();


      ctx.font = "20px CustomFont";
      ctx.fillStyle = "black";
      ctx.fillText(String(this.traj+1) + "/" + String(this.nTrajs+1),480-40-30-30+70-5+20+10,85+170);
      ctx.fill();

      //350*520
      // console.log(window.qdWidth);
      // console.log(window.qdHeight);
      // console.log("\n");
      // 1296
      // 696
      //
      // ctx.drawImage(
      //   this.img1,
      //   200-100-100,
      //   0,
      //   0.3105*window.qdWidth,
      //   0.859*window.qdHeight
      // );
      //
      // ctx.drawImage(
      //   this.img2,
      //   290+this.offset-100-170+70,
      //   0,
      //   0.3105*window.qdWidth,
      //   0.859*window.qdHeight
      // );

      ctx.drawImage(
        this.img1,
        200-100-100,
        0,
        402.5,
        598
      );

      ctx.drawImage(
        this.img2,
        290+this.offset-100-170+70,
        0,
        402.5,
        598
      );

      this.drawButton(ctx,this.b1_x,this.b1_y,this.b1_w,this.b1_h,this.b1Color,"Left",this.b1_x-(this.b1_w/2)+50,this.b1_y+(this.b1_h/2)+5);
      this.drawButton(ctx,this.b2_x,this.b1_y,this.b1_w,this.b1_h,this.b2Color,"Right",this.b2_x-(this.b1_w/2)+42,this.b1_y+(this.b1_h /2)+5)


      this.drawButton(ctx,this.b3_x,  this.b3_y,this.b3_h,this.b3_w,this.b3Color,"Same",this.b3_x+(this.b3_h/2)-25,  this.b3_y+(this.b3_w/2)+5);
      this.drawButton(ctx,this.b3_x,this.b4_y,this.b3_h,this.b3_w,this.b4Color,"Can't Tell",  this.b3_x+(this.b3_h/2)-50,this.b4_y+(this.b3_w/2)+5);

    }



  }
}
