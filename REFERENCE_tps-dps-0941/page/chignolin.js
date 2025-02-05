document.addEventListener("DOMContentLoaded", function () {
  stage_chignolin = new NGL.Stage("viewport-chignolin");
  stage_chignolin.setParameters({ backgroundColor: "white" });
  stage_chignolin
    .loadFile("data/chignolin/path.gro", {
      defaultRepresentation: true,
      asTrajectory: true,
    })
    .then(function (comp) {
      comp.setName("simulation-chignolin");
      comp.addRepresentation("ball+stick", {
        aspectRatio: 1.0,
        radius: 0.15,
      });
      comp.addRepresentation("distance", {
        atomPair: [[37, 76]],
        color: "#FEC220",
        linewidth: 12.0,
        useCylinder: true,
        radius: 0.3,
        label_size: 0,
      });
      comp.addRepresentation("distance", {
        atomPair: [[30, 95]],
        color: "#3FA796",
        linewidth: 12.0,
        useCylinder: true,
        radius: 0.3,
        labelSize: 0,
      });
      comp.addTrajectory();
    });

  var toggleSpin_chignolin = document.getElementById(
    "toggleSpin-chignolin"
  );
  var isSpinning_chignolin = false;
  toggleSpin_chignolin.addEventListener("click", function () {
    if (!isSpinning_chignolin) {
      stage_chignolin.setSpin([0, 1, 0], 0.01);
      isSpinning_chignolin = true;
    } else {
      stage_chignolin.setSpin(null, null);
      isSpinning_chignolin = false;
    }
  });

  var toggleRunMDs_chignolin = document.getElementById(
    "toggleRunMDs-chignolin"
  );
  var isRunning_chignolin = false;
  toggleRunMDs_chignolin.addEventListener("click", function () {
    var trajComp_chignolin = stage_chignolin.getComponentsByName(
      "simulation-chignolin"
    ).list[0].trajList[0];
    var player_chignolin = new NGL.TrajectoryPlayer(
      trajComp_chignolin.trajectory,
      { step: 80, mode: "once", interpolateType: "spline", timeout: 200 }
    );
    if (!isRunning_chignolin) {
      player_chignolin.play();
      isRunning_chignolin = true;
    } else {
      player_chignolin.play();
      player_chignolin.pause();
      isRunning_chignolin = false;
    }
  });

  stage_chignolin_initial = new NGL.Stage("viewport-chignolin-initial");
  stage_chignolin_initial.setParameters({ backgroundColor: "white" });
  stage_chignolin_initial.loadFile("data/chignolin/unfolded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
    comp.addRepresentation("distance", {
      atomPair: [[37, 76]],
      color: "#FEC220",
      linewidth: 12.0,
      useCylinder: true,
      radius: 0.3,
      label_size: 0,
    });
    comp.addRepresentation("distance", {
      atomPair: [[30, 95]],
      color: "#3FA796",
      linewidth: 12.0,
      useCylinder: true,
      radius: 0.3,
      labelSize: 0,
    });
  });

  var toggleSpin_chignolin_initial = document.getElementById("toggleSpin-chignolin-inital");
  var isSpinning_chignolin_initial = false;
  toggleSpin_chignolin_initial.addEventListener("click", function () {
    if (!isSpinning_chignolin_initial) {
      stage_chignolin_initial.setSpin([0, 1, 0], 0.01);
      isSpinning_chignolin_initial = true;
    } else {
      stage_chignolin_initial.setSpin(null, null);
      isSpinning_chignolin_initial = false;
    }
  });

  stage_chignolin_target = new NGL.Stage("viewport-chignolin-target");
  stage_chignolin_target.setParameters({ backgroundColor: "white" });
  stage_chignolin_target.loadFile("data/chignolin/folded.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      aspectRatio: 1.0,
      radius: 0.15,
    });
    comp.addRepresentation("distance", {
      atomPair: [[37, 76]],
      color: "#FEC220",
      linewidth: 12.0,
      useCylinder: true,
      radius: 0.3,
      label_size: 0,
    });
    comp.addRepresentation("distance", {
      atomPair: [[30, 95]],
      color: "#3FA796",
      linewidth: 12.0,
      useCylinder: true,
      radius: 0.3,
      labelSize: 0,
    });
  });

  var toggleSpin_chignolin_target = document.getElementById("toggleSpin-chignolin-target")
  var isSpinning_chignolin_target = false;
  toggleSpin_chignolin_target.addEventListener("click", function () {
    if (!isSpinning_chignolin_target) {
      stage_chignolin_target.setSpin([0, 1, 0], 0.01);
      isSpinning_chignolin_target = true;
    } else {
      stage_chignolin_target.setSpin(null, null);
      isSpinning_chignolin_target = false;
    }
  });
});