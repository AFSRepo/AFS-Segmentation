#!/bin/bash

VERSION="0.0.0 test"

# trap keyboard interrupt (control-c)
trap control_c SIGINT

function setPath {
    cat <<SETPATH

--------------------------------------------------------------------------------------
Error locating ANTS
--------------------------------------------------------------------------------------
It seems that the ANTSPATH environment variable is not set. Please add the ANTSPATH
variable. This can be achieved by editing the .bash_profile in the home directory.
Add:

ANTSPATH=/home/yourname/bin/ants/

Or the correct location of the ANTS binaries.

Alternatively, edit this script ( `basename $0` ) to set up this parameter correctly.

SETPATH
    exit 1
}

# Uncomment the line below in case you have not set the ANTSPATH variable in your environment.
# export ANTSPATH=${ANTSPATH:="$HOME/bin/ants/"} # EDIT THIS

#ANTSPATH=YOURANTSPATH
if [[ ${#ANTSPATH} -le 3 ]];
  then
    setPath >&2
  fi

ANTS=${ANTSPATH}/antsRegistration

if [[ ! -s ${ANTS} ]];
  then
    echo "antsRegistration program can't be found. Please (re)define \$ANTSPATH in your environment."
    exit
  fi

function Usage {
    cat <<USAGE

Usage:

`basename $0` -d ImageDimension -f FixedImage -m MovingImage -o OutputPrefix

Compulsory arguments:

     -d:  ImageDimension: 2 or 3 (for 2 or 3 dimensional registration of single volume)

     -f:  Fixed image or source image or reference image

     -m:  Moving image or target image

     -o:  OutputPrefix: A prefix that is prepended to all output files.

Optional arguments:

     -n:  Number of threads (default = 1)

     -t:  transform type (default = 's')
         t: translation
         k: similarity
         r: rigid
         a: rigid + affine
         s: rigid + affine + deformable syn
         b: rigid + affine + deformable b-spline syn
         d: similarity + deformable b-spline syn
         v: similarity + time varying velocity field
         g: rigid + similarity + deformable b-spline syn (fast)
         y: similarity + deformable b-spline syn (fast)
         h: rigid + similarity

     -r:  radius for cross correlation metric used during SyN stage (default = 4)

     -s:  spline distance for deformable B-spline SyN transform (default = 26)

     -p:  precision type (default = 'd')
        f: float
        d: double

     -j:  use histogram matching (default = 0)
        0: false
        1: true

     NB:  Multiple image pairs can be specified for registration during the SyN stage.
          Specify additional images using the '-m' and '-f' options.  Note that image
          pair correspondence is given by the order specified on the command line.
          Only the first fixed and moving image pair is used for the linear resgitration
          stages.

Example:

`basename $0` -d 3 -f fixedImage.nii.gz -m movingImage.nii.gz -o output

--------------------------------------------------------------------------------------
ANTs was created by:
--------------------------------------------------------------------------------------
Brian B. Avants, Nick Tustison and Gang Song
Penn Image Computing And Science Laboratory
University of Pennsylvania

script by Nick Tustison

USAGE
    exit 1
}

function Help {
    cat <<HELP

Usage:

`basename $0` -d ImageDimension -f FixedImage -m MovingImage -o OutputPrefix

Example Case:

`basename $0` -d 3 -f fixedImage.nii.gz -m movingImage.nii.gz -o output

Compulsory arguments:

     -d:  ImageDimension: 2 or 3 (for 2 or 3 dimensional registration of single volume)

     -f:  Fixed image or source image or reference image

     -m:  Moving image or target image

     -o:  OutputPrefix: A prefix that is prepended to all output files.

Optional arguments:

     -n:  Number of threads (default = 1)

     -t:  transform type (default = 's')
        t: translation
        k: similarity
        r: rigid
        a: rigid + affine
        s: rigid + affine + deformable syn
        b: rigid + affine + deformable b-spline syn
        d: similarity + deformable b-spline syn
        v: similarity + time varying velocity field
        g: rigid + similarity + deformable b-spline syn (fast)
        y: similarity + deformable b-spline syn (fast)
        h: rigid + similarity

     -r:  radius for cross correlation metric used during SyN stage (default = 4)

     -s:  spline distance for deformable B-spline SyN transform (default = 26)

     -p:  precision type (default = 'd')
        f: float
        d: double

     -j:  use histogram matching (default = 0)
        0: false
        1: true

     NB:  Multiple image pairs can be specified for registration during the SyN stage.
          Specify additional images using the '-m' and '-f' options.  Note that image
          pair correspondence is given by the order specified on the command line.
          Only the first fixed and moving image pair is used for the linear resgitration
          stages.

--------------------------------------------------------------------------------------
Get the latest ANTs version at:
--------------------------------------------------------------------------------------
https://github.com/stnava/ANTs/

--------------------------------------------------------------------------------------
Read the ANTS documentation at:
--------------------------------------------------------------------------------------
http://stnava.github.io/ANTs/

--------------------------------------------------------------------------------------
ANTS was created by:
--------------------------------------------------------------------------------------
Brian B. Avants, Nick Tustison and Gang Song
Penn Image Computing And Science Laboratory
University of Pennsylvania

Relevent references for this script include:
   * http://www.ncbi.nlm.nih.gov/pubmed/20851191
   * http://www.frontiersin.org/Journal/10.3389/fninf.2013.00039/abstract
--------------------------------------------------------------------------------------
script by Nick Tustison
--------------------------------------------------------------------------------------

HELP
    exit 1
}

function reportMappingParameters {
    cat <<REPORTMAPPINGPARAMETERS

--------------------------------------------------------------------------------------
 Mapping parameters
--------------------------------------------------------------------------------------
 ANTSPATH is $ANTSPATH

 Dimensionality:           $DIM
 Output name prefix:       $OUTPUTNAME
 Fixed images:             ${FIXEDIMAGES[@]}
 Moving images:            ${MOVINGIMAGES[@]}
 Number of threads:        $NUMBEROFTHREADS
 Spline distance:          $SPLINEDISTANCE
 Transform type:           $TRANSFORMTYPE
 CC radius:                $CCRADIUS
 Precision:                $PRECISIONTYPE
 Use histogram matching    $USEHISTOGRAMMATCHING
======================================================================================
REPORTMAPPINGPARAMETERS
}

cleanup()
# example cleanup function
{

  cd ${currentdir}/

  echo "\n*** Performing cleanup, please wait ***\n"

# 1st attempt to kill all remaining processes
# put all related processes in array
runningANTSpids=( `ps -C antsRegistration | awk '{ printf "%s\n", $1 ; }'` )

# debug only
  #echo list 1: ${runningANTSpids[@]}

# kill these processes, skip the first since it is text and not a PID
for (( i = 1; i < ${#runningANTSpids[@]}; i++ ))
  do
    echo "killing:  ${runningANTSpids[${i}]}"
    kill ${runningANTSpids[${i}]}
done

  return $?
}

control_c()
# run if user hits control-c
{
  echo -en "\n*** User pressed CTRL + C ***\n"
  cleanup
  exit $?
  echo -en "\n*** Script cancelled by user ***\n"
}


# Provide output for Help
if [[ "$1" == "-h" || $# -eq 0 ]];
  then
    Help >&2
  fi

#################
#
# default values
#
#################

DIM=3
FIXEDIMAGES=()
MOVINGIMAGES=()
OUTPUTNAME=output
NUMBEROFTHREADS=1
SPLINEDISTANCE=26
TRANSFORMTYPE='s'
PRECISIONTYPE='d'
USEHISTOGRAMMATCHING=0
#CCRADIUS=3
CCRADIUS=5
CONVWIN=10

# reading command line arguments
while getopts "d:f:h:j:m:n:o:p:r:s:t:" OPT
  do
  case $OPT in
      h) #help
   Help
   exit 0
   ;;
      d)  # dimensions
   DIM=$OPTARG
   ;;
      f)  # fixed image
   FIXEDIMAGES[${#FIXEDIMAGES[@]}]=$OPTARG
   ;;
      j)  # histogram matching
   USEHISTOGRAMMATCHING=$OPTARG
   ;;
      m)  # moving image
   MOVINGIMAGES[${#MOVINGIMAGES[@]}]=$OPTARG
   ;;
      n)  # number of threads
   NUMBEROFTHREADS=$OPTARG
   ;;
      o) #output name prefix
   OUTPUTNAME=$OPTARG
   ;;
      p)  # precision type
   PRECISIONTYPE=$OPTARG
   ;;
      r)  # cc radius
   CCRADIUS=$OPTARG
   ;;
      s)  # spline distance
   SPLINEDISTANCE=$OPTARG
   ;;
      t)  # transform type
   TRANSFORMTYPE=$OPTARG
   ;;
     \?) # getopts issues an error message
   echo "$USAGE" >&2
   exit 1
   ;;
  esac
done

###############################
#
# Check inputs
#
###############################
if [[ ${#FIXEDIMAGES[@]} -ne ${#MOVINGIMAGES[@]} ]];
  then
    echo "Number of fixed images is not equal to the number of moving images."
    exit 1
  fi

for(( i=0; i<${#FIXEDIMAGES[@]}; i++ ))
  do
    if [[ ! -f "${FIXEDIMAGES[$i]}" ]];
      then
        echo "Fixed image '${FIXEDIMAGES[$i]}' does not exist.  See usage: '$0 -h 1'"
        exit 1
      fi
    if [[ ! -f "${MOVINGIMAGES[$i]}" ]];
      then
        echo "Moving image '${MOVINGIMAGES[$i]}' does not exist.  See usage: '$0 -h 1'"
        exit 1
      fi
  done

###############################
#
# Set number of threads
#
###############################

ORIGINALNUMBEROFTHREADS=${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS}
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NUMBEROFTHREADS
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS

##############################
#
# Print out options
#
##############################

reportMappingParameters

##############################
#
# Infer the number of levels based on
# the size of the input fixed image.
#
##############################

ISLARGEIMAGE=0

SIZESTRING=$( ${ANTSPATH}/PrintHeader ${FIXEDIMAGES[0]} 2 )
SIZESTRING="${SIZESTRING%\\n}"
SIZE=( `echo $SIZESTRING | tr 'x' ' '` )

for (( i=0; i<${#SIZE[@]}; i++ ))
  do
    if [[ ${SIZE[$i]} -gt 256 ]];
      then
        ISLARGEIMAGE=1
        break
      fi
  done

##############################
#
# Construct mapping stages
#
##############################

TRANSLATIONCONVERGENCE="[1000x500x500x500,1e-7,10]"
TRANSLATIONSHRINKFACTORS="8x4x2x1"
TRANSLATIONSMOOTHINGSIGMAS="3x2x1x0vox"

RIGIDCONVERGENCE="[1000x500x250x250,1e-6,10]"
RIGIDSHRINKFACTORS="8x4x2x1"
RIGIDSMOOTHINGSIGMAS="3x2x1x0vox"

AFFINECONVERGENCE="[1000x500x250x250,1e-6,10]"
AFFINESHRINKFACTORS="8x4x2x1"
AFFINESMOOTHINGSIGMAS="3x2x1x0vox"

SIMILARITYCONVERGENCE="[1000x500x250x250,1e-6,10]"
SIMILARITYSHRINKFACTORS="8x4x2x1"
SIMILARITYSMOOTHINGSIGMAS="3x2x1x0vox"

#SYNCONVERGENCEFAST="[500x300x150x250x0,1e-6,30]"
SYNCONVERGENCEFAST="[500x300x300x300x0,1e-6,10]"
SYNCONVERGENCE="[500x500x500x500x500,1e-7,10]"
#SYNCONVERGENCE="[500x500x500x500x500,1e-7,15]"
SYNSHRINKFACTORS="5x4x3x2x1"
SYNSMOOTHINGSIGMAS="4x3x2x1x0vox"

if [[ $ISLARGEIMAGE -eq 1 ]];
  then
    TRANSLATIONCONVERGENCE="[1000x500x500x500,1e-7,20]"
    TRANSLATIONSHRINKFACTORS="12x8x4x2"
    TRANSLATIONSMOOTHINGSIGMAS="4x3x2x1vox"

    RIGIDCONVERGENCE="[1000x500x250x250,1e-6,10]"
    RIGIDSHRINKFACTORS="12x8x4x2"
    RIGIDSMOOTHINGSIGMAS="4x3x2x1vox"

    AFFINECONVERGENCE="[1000x500x250x250,1e-6,10]"
    AFFINESHRINKFACTORS="8x4x2x1"
    AFFINESMOOTHINGSIGMAS="3x2x1x0vox"

    SIMILARITYCONVERGENCE="[1000x500x250x250,1e-6,10]"
    SIMILARITYSHRINKFACTORS="8x4x2x1"
    SIMILARITYSMOOTHINGSIGMAS="3x2x1x0vox"

    SYNCONVERGENCEFAST="[500x300x300x0,1e-6,10]"
    SYNCONVERGENCE="[500x500x500x500,1e-7,10]"
    SYNSHRINKFACTORS="4x3x2x1"
    SYNSMOOTHINGSIGMAS="3x2x1x0vox"
  fi

TRANSLATIONSTAGE="--initial-moving-transform [${FIXEDIMAGES[0]},${MOVINGIMAGES[0]},1] \
                  --transform Translation[0.1] \
                  --metric MI[${FIXEDIMAGES[0]},${MOVINGIMAGES[0]},1,32,Regular,0.25] \
                  --convergence $TRANSLATIONCONVERGENCE \
                  --shrink-factors $TRANSLATIONSHRINKFACTORS \
                  --smoothing-sigmas $TRANSLATIONSMOOTHINGSIGMAS"

RIGIDSTAGE="--initial-moving-transform [${FIXEDIMAGES[0]},${MOVINGIMAGES[0]},1] \
            --transform Rigid[0.1] \
            --metric MI[${FIXEDIMAGES[0]},${MOVINGIMAGES[0]},1,32,Regular,0.25] \
            --convergence $RIGIDCONVERGENCE \
            --shrink-factors $RIGIDSHRINKFACTORS \
            --smoothing-sigmas $RIGIDSMOOTHINGSIGMAS"

AFFINESTAGE="--transform Affine[0.1] \
             --metric MI[${FIXEDIMAGES[0]},${MOVINGIMAGES[0]},1,32,Regular,0.25] \
             --convergence $AFFINECONVERGENCE \
             --shrink-factors $AFFINESHRINKFACTORS \
             --smoothing-sigmas $AFFINESMOOTHINGSIGMAS"


SIMILARITYSTAGE="--transform Similarity[0.1] \
                 --metric MI[${FIXEDIMAGES[0]},${MOVINGIMAGES[0]},1,32,Regular,0.25] \
                 --convergence $SIMILARITYCONVERGENCE \
                 --shrink-factors $SIMILARITYSHRINKFACTORS \
                 --smoothing-sigmas $SIMILARITYSMOOTHINGSIGMAS"

if [[ $TRANSFORMTYPE == 'y' ]]
  then
    SIMILARITYSTAGE="--restrict-deformation 0x0x0x1x1x1x1 \
                    ${SIMILARITYSTAGE}"
  fi

if [[ $TRANSFORMTYPE == 'k' || $TRANSFORMTYPE == 'd' || $TRANSFORMTYPE == 'v' || $TRANSFORMTYPE == 'y' ]];
  then
    SIMILARITYSTAGE="--initial-moving-transform [${FIXEDIMAGES[0]},${MOVINGIMAGES[0]},1] \
                    ${SIMILARITYSTAGE}"
  fi

SYNMETRICS=''
for(( i=0; i<${#FIXEDIMAGES[@]}; i++ ))
  do
    SYNMETRICS="$SYNMETRICS --metric CC[${FIXEDIMAGES[$i]},${MOVINGIMAGES[$i]},1,${CCRADIUS}]"
    #SYNMETRICS="$SYNMETRICS --metric MI[${FIXEDIMAGES[$i]},${MOVINGIMAGES[$i]},1,32,Regular,0.25]"
    #SYNMETRICS="$SYNMETRICS --metric MeanSquares[${FIXEDIMAGES[$i]},${MOVINGIMAGES[$i]},1,0] \
    #            $SYNMETRICS --metric MI[${FIXEDIMAGES[$i]},${MOVINGIMAGES[$i]},1,32,Regular,0.25]"
done

SYNSTAGE="${SYNMETRICS} \
          --convergence $SYNCONVERGENCE \
          --shrink-factors $SYNSHRINKFACTORS \
          --smoothing-sigmas $SYNSMOOTHINGSIGMAS"

if [[ $TRANSFORMTYPE == 'g' || $TRANSFORMTYPE == 'y' ]];
  then
    SYNSTAGE="${SYNMETRICS} \
              --convergence $SYNCONVERGENCEFAST \
              --shrink-factors $SYNSHRINKFACTORS \
              --smoothing-sigmas $SYNSMOOTHINGSIGMAS"
  fi

if [[ $TRANSFORMTYPE == 'v' ]];
  then
    SYNSTAGE="--transform TimeVaryingVelocityField[0.1,8,1,0.0,0.05,0] \
        $SYNSTAGE"
  else
    SYNSTAGE="--transform BSplineSyN[0.1,${SPLINEDISTANCE},0,3] \
              $SYNSTAGE"
  fi

STAGES=''
case "$TRANSFORMTYPE" in
"t")
  STAGES="$TRANSLATIONSTAGE"
  ;;
"k")
  STAGES="$SIMILARITYSTAGE"
  ;;
"r")
  STAGES="$RIGIDSTAGE"
  ;;
"a")
  STAGES="$RIGIDSTAGE $AFFINESTAGE"
  ;;
"b" | "s")
  STAGES="$RIGIDSTAGE $AFFINESTAGE $SYNSTAGE"
  ;;
"d" | "v")
  STAGES="$SIMILARITYSTAGE $SYNSTAGE"
  ;;
"g")
  STAGES="$RIGIDSTAGE $SIMILARITYSTAGE $SYNSTAGE"
  ;;
"y")
  STAGES="$SIMILARITYSTAGE $SYNSTAGE"
  ;;
"h")
  STAGES="$RIGIDSTAGE $SIMILARITYSTAGE"
  ;;
*)
  echo "Transform type '$TRANSFORMTYPE' is not an option.  See usage: '$0 -h 1'"
  exit
  ;;
esac

PRECISION=''
case "$PRECISIONTYPE" in
"f")
  PRECISION="--float 1"
  ;;
"d")
  PRECISION="--float 0"
  ;;
*)
  echo "Precision type '$PRECISIONTYPE' is not an option.  See usage: '$0 -h 1'"
  exit
  ;;
esac

COMMAND="${ANTS} --verbose 1 \
                 --dimensionality $DIM $PRECISION \
                 --output [$OUTPUTNAME,${OUTPUTNAME}Warped.nii.gz] \
                 --interpolation BSpline \
                 --winsorize-image-intensities [0.005,0.995] \
                 --use-histogram-matching ${USEHISTOGRAMMATCHING} \
                 $STAGES"

echo " antsRegistration call:"
echo "--------------------------------------------------------------------------------------"
echo ${COMMAND}
echo "--------------------------------------------------------------------------------------"

$COMMAND

###############################
#
# Restore original number of threads
#
###############################

ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$ORIGINALNUMBEROFTHREADS
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
