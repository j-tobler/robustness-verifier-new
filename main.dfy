include "IO/FileIO.dfy"
include "string_utils.dfy"
include "lipschitz.dfy"
include "basic_arithmetic.dfy"

module MainModule {
  import FileIO
  import StringUtils
  import opened Lipschitz
  import BasicArithmetic

  method Main(args: seq<string>)
    decreases *
  {
    // parse command line arguments
    if |args| != 3 {
      print "Usage: main <neural_network_input.txt> <GRAM_ITERATIONS>\n";
      print "Usage: main <neural_network_input.txt> <lipshitz_bounds_input.txt>\n";      
      return;
    }

    // Parse neural network from file (unverified).
    var neuralNetStr: string := ReadFromFile(args[1]);
    var maybeNeuralNet: (bool, NeuralNetwork) := ParseNeuralNet(neuralNetStr);
    expect maybeNeuralNet.0, "Failed to parse neural network.";
    var neuralNet: NeuralNetwork := maybeNeuralNet.1;

    var lipBounds: seq<seq<real>>;
    print "[\n";
    
    if StringUtils.IsInt(args[2]) {
      var GRAM_ITERATIONS: int := StringUtils.ParseInt(args[2]);
      if GRAM_ITERATIONS <= 0 {
        print "<GRAM_ITERATIONS> should be positive";
        return;
      }
      
      var L := new LipschitzBounds(GRAM_ITERATIONS);
    
      /* ===================== Generate Lipschitz bounds ===================== */
      // Generate spectral norms for the matrices comprising the neural net.
      // We currently assume an external implementation for generating these.
      var specNorms: seq<real> := GenerateSpecNorms(L, neuralNet);
      // Generate the Lipschitz bounds for each logit in the output vector.
      lipBounds := GenMarginLipBounds(L, neuralNet, specNorms);

      print "{\n";
      print "  \"lipschitz_bounds\": ", lipBounds, ",\n";
      print "  \"GRAM_ITERATIONS\": ", GRAM_ITERATIONS, ",\n";
      print "  \"provenance\": \"generated\"\n";
      print "}\n";      
    }else{
      print "Reading margin lipschitz bounds from a file not (currently) supported\n";
      return;

      /*
      var lipBoundsStr: string := ReadFromFile(args[2]);
      var lb: seq<string> := StringUtils.Split(lipBoundsStr,'\n');
      var line: string := lb[0];
      var realsStr: seq<string> := StringUtils.Split(line,',');
      var areReals: bool := AreReals(realsStr);
      if !areReals {
        print "Error: Lipschitz bounds is not a vector.\n";
        return;
      }
      lipBounds := ParseReals(realsStr);
      if |lipBounds| != |neuralNet[|neuralNet|-1]| {
        print "Given Lipschitz bounds vector not compatible with neural network: \n";
	print    "Lipschitz bounds vector has size: ", |lipBounds|, "\n";
	return;
      }
      // FIXME: check bounds are positive and remove the first axiom
      assume {:axiom} forall i | 0 <= i < |lipBounds| :: 0.0 <= lipBounds[i];
      assume {:axiom} AreLipBounds(neuralNet, lipBounds);
      print "{\n";
      print "  \"lipschitz_bounds\": ", lipBounds, ",\n";
      print "  \"provenance\": \"loaded from file\"\n";
      print "}\n";
      */
    }

    /* ================= Repeatedly certify output vectors ================= */

    var inputStr: string := ReadFromFile("/dev/stdin");

    // Extract output vector and error margin, which are space-separated.
    var lines: seq<string> := StringUtils.Split(inputStr, '\n');
    
    if |lines| <= 0 {
      return;
    }

    var l := 0;
    while l < |lines|
      decreases |lines| - l
    {
      var line := lines[l];
      l := l + 1;
      var inputSeq: seq<string> := StringUtils.Split(line, ' ');
      if |inputSeq| != 2 {
        // as soon as we see bad input, stop silently so that the end of the input won't cause junk to be printed
	print "]\n";
	return;
      }
      
      // Parse output vector.
      if inputSeq[0] == "" {
	print "]\n";      
        return;
      }
      
      var realsStr: seq<string> := StringUtils.Split(inputSeq[0], ',');
      var areReals: bool := AreReals(realsStr);
      if !areReals {
        print "Error: The given output vector contained non-real values.\n";
        continue;
      }
      var outputVector := ParseReals(realsStr);
      
      // Parse error margin.
      if inputSeq[1] == "" {
        print "Error: The given error margin was found to be empty.\n";
        continue;
      }
      var isReal: bool := StringUtils.IsReal(inputSeq[1]);
      if !isReal {
        print "Error: The given error margin is not of type real.\n";
        continue;
      }
      var errorMargin := StringUtils.ParseReal(inputSeq[1]);

      /* ======================= Certify Robustness ======================== */

      // The given output vector must be compatible with the neural network.
      if |outputVector| != |lipBounds| {
        print "Error: Expected a vector of size ", |lipBounds|,
          ", but got ", |outputVector|, ".\n";
        continue;
      }
      // Use the generated Lipschitz bounds to certify robustness.
      var robust: bool := Certify(outputVector, errorMargin, lipBounds);
      /* Verification guarantees that 'true' is only printed when for all input
      vectors v where applying the neural network to v results in the given
      output vector, this input-output pair of vectors is robust with respect
      to the given error margin. */
      assert robust ==> forall v: Vector |
        CompatibleInput(v, neuralNet) && NN(neuralNet, v) == outputVector ::
        Robust(v, outputVector, errorMargin, neuralNet);

      print ",\n";
      print "{\n";
      print "\"output\": ";
      print outputVector, ",\n";
      print "\"radius\": ";
      print errorMargin, ",\n";
      print "\"certified\": ";
      print robust, "\n";
      print "}\n";
    }
    print "]\n";
  }

  method ParseNeuralNet(xs: string) returns (t: (bool, NeuralNetwork))
    // Todo: Verify
  {
    var err: string := "";
    var matrices: seq<Matrix> := [];
    var i := 0;
    while i < |xs| {
      // Expect matrix
      if i >= |xs| - 1 || xs[i..i+2] != "[[" {
        print "One";
        return (false, [[[0.0]]]);
      }
      var j := i + 2;
      while xs[j-2..j] != "]]"
        invariant j <= |xs|
        decreases |xs| - j
      {
        if j >= |xs| {
          print "Two";
          return (false, [[[0.0]]]);
        }
        j := j + 1;
      }
      // xs[i..j] == "[[...],...,[...]]"
      var ys := xs[i+1..j-1];
      // ys == "[...],...,[...]"
      var k := 0;
      var vectors: seq<Vector> := [];
      while k < |ys| {
        // Expect vector
        if ys[k] != '[' {
          print "Three";
          return (false, [[[0.0]]]);
        }
        var l := k;
        while ys[l] != ']'
          invariant l < |ys|
          decreases |ys| - l
        {
          if l + 1 >= |ys| {
            print "Four";
            return (false, [[[0.0]]]);
          }
          l := l + 1;
        }
        // ys[k..l] == "[r1,r2,...,rn"
        var zs := ys[k+1..l];
        // zs == "r1,r2,...,rn"
        var realsStr: seq<string> := StringUtils.Split(zs, ',');
        var areReals: bool := AreReals(realsStr);
        if !areReals {
          print "Five\n";
          return (false, [[[0.0]]]);
        }
        var v: seq<real> := ParseReals(realsStr);
        if |v| == 0 {
          return (false, [[[0.0]]]);
        }
        var v': Vector := v;
        vectors := vectors + [v'];
        k := l + 2; // skip comma
      }
      var matrixWellFormed := IsMatrixWellFormed(vectors);
      if !matrixWellFormed {
        print "Six";
        return (false, [[[0.0]]]);
      }
      var matrix: Matrix := vectors;
      matrices := matrices + [Transpose(matrix)]; // need to transpose for comptability with python output
      i := j + 1; // xs[j] == ',' or EOF
    }
    var neuralNetWellFormed := IsNeuralNetWellFormed(matrices);
    if !neuralNetWellFormed {
      print "Seven\n";
      print |matrices|, "\n";
      if |matrices| == 2 {
        print |matrices[0]|, "\n";
        if |matrices[0]| > 0 {
          print |matrices[0][0]|, "\n";
        }
        print |matrices[1]|, "\n";
        if |matrices[1]| > 0 {
          print |matrices[1][0]|, "\n";
        }
      }
      return (false, [[[0.0]]]);
    }
    var neuralNet: NeuralNetwork := matrices;
    return (true, neuralNet);
  }

  method IsNeuralNetWellFormed(n: seq<Matrix>) returns (b: bool)
    ensures b ==>
      |n| > 0 &&
      forall i: int :: 0 <= i < |n| - 1 ==> |n[i]| == |n[i + 1][0]|
  {
    if |n| == 0 {
      return false;
    }
    var i := 0;
    while i < |n| - 1
      invariant 0 <= i <= |n| - 1
      invariant forall j | 0 <= j < i :: |n[j]| == |n[j + 1][0]|
    {
      if |n[i]| != |n[i + 1][0]| {
        return false;
      }
      i := i + 1;
    }
    return true;
  }

  method IsMatrixWellFormed(m: seq<seq<real>>) returns (b: bool)
    ensures b ==>
      |m| > 0 &&
      |m[0]| > 0 &&
      forall i, j: int :: 0 <= i < |m| && 0 <= j < |m| ==> |m[i]| == |m[j]|
  {
    if |m| == 0 || |m[0]| == 0 {
      return false;
    }
    var size := |m[0]|;
    var i := 1;
    while i < |m|
      invariant 0 <= i <= |m|
      invariant forall j | 0 <= j < i :: |m[j]| == size
    {
      if |m[i]| != size {
        return false;
      }
      i := i + 1;
    }
    return true;
  }


  method AreReals(realsStr: seq<string>) returns (b: bool)
    ensures b ==> forall i | 0 <= i < |realsStr| :: StringUtils.IsReal(realsStr[i])
  {
    for i := 0 to |realsStr|
      invariant forall j | 0 <= j < i :: StringUtils.IsReal(realsStr[j])
    {
      var isReal := StringUtils.IsReal(realsStr[i]);
      if !isReal {
        print realsStr[i];
        print "\n";
        return false;
      }
    }
    return true;
  }

  method ParseReals(realsStr: seq<string>) returns (reals: seq<real>)
    requires forall i | 0 <= i < |realsStr| :: StringUtils.IsReal(realsStr[i])
  {
    reals := [];
    for i := 0 to |realsStr| {
      var r := StringUtils.ParseReal(realsStr[i]);
      reals := reals + [r];
    }
  }

  method ReadFromFile(filename: string) returns (str: string) {
    var readResult := FileIO.ReadBytesFromFile(filename);
    expect readResult.Success?, "Unexpected failure reading from " +
      filename + ": " + readResult.error;
    str := seq(|readResult.value|,
      i requires 0 <= i < |readResult.value| => readResult.value[i] as char);
  }
}
