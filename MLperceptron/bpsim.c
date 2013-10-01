/*
 * See bottom for address of author.
 *
 * title:       bpsim.c
 * author:      Josiah C. Hoskins
 * date:        June 1987
 *
 * purpose:     backpropagation learning rule neural net simulator
 *              for the tabula rasa Little Red Riding Hood example
 *
 * description: Bpsim provides an implementation of a neural network
 *              containing a single hidden layer which uses the
 *              generalized backpropagation delta rule for learning.
 *              A simple user interface is supplied for experimenting
 *              with a neural network solution to the Little Red Riding
 *              Hood example described in the text.
 *
 *              In addition, bpsim contains some useful building blocks
 *              for further experimentation with single layer neural
 *              networks. The data structure which describes the general
 *              processing unit allows one to easily investigate different
 *              activation (output) and/or error functions. The utility
 *              function create_link can be used to create links between
 *              any two units by supplying your own create_in_out_links
 *              function. The flexibility of creating units and links
 *              to your specifications allows one to modify the code
 *              to tune the network architecture to problems of interest.
 *
 *              There are some parameters that perhaps need some
 *              explanation. You will notice that the target values are
 *              either 0.1 or 0.9 (corresponding to the binary values
 *              0 or 1). With the sigmoidal function used in out_f the
 *              weights become very large if 0 and 1 are used as targets.
 *              The ON_TOLERANCE value is used as a criteria for an output
 *              value to be considered "on", i.e., close enough to the
 *              target of 0.9 to be considered 1. The learning_rate and
 *              momentum variables may be changed to vary the rate of
 *              learning, however, in general they each should be less
 *              than 1.0.
 *
 *              Bpsim has been compiled using CI-C86 version 2.30 on an
 *              IBM-PC and the Sun C compiler on a Sun 3/160.
 *
 *              Note to compile and link on U*IX machines use:
 *                      cc -o bpsim bpsim.c -lm
 *
 *              For other machines remember to link in the math library.
 *
 * status:      This program may be freely used, modified, and distributed
 *              except for commercial purposes.
 *
 * Copyright (c) 1987   Josiah C. Hoskins
 */
 /* Modified to function properly under Turbo C by replacing malloc(...)
    with calloc(...,1). Thanks to Pavel Rozalski who detected the error.
    He assumed that Turbo C's "malloc" doesn't automatically set pointers
    to NULL - and he was right!
    Thomas Muhr, Berlin April, 1988
 */

#include <math.h>
#include <stdio.h>
#include <ctype.h>

#define BUFSIZ          512

#define FALSE           0
#define TRUE            !FALSE
#define NUM_IN          6       /* number of input units */
#define NUM_HID         3       /* number of hidden units */
#define NUM_OUT         7       /* number of output units */
#define TOTAL           (NUM_IN + NUM_HID + NUM_OUT)
#define BIAS_UID        (TOTAL) /* threshold unit */

/* macros to provide indexes for processing units */
#define IN_UID(X)       (X)
#define HID_UID(X)      (NUM_IN + X)
#define OUT_UID(X)      (NUM_IN + NUM_HID + X)
#define TARGET_INDEX(X) (X - (NUM_IN + NUM_HID))

#define WOLF_PATTERN    0
#define GRANDMA_PATTERN 1
#define WOODCUT_PATTERN 2
#define PATTERNS        3       /* number of input patterns */
#define ERROR_TOLERANCE 0.01
#define ON_TOLERANCE    0.8     /* a unit's output is on if > ON_TOLERENCE */
#define NOTIFY          10      /* iterations per dot notification */
#define DEFAULT_ITER    250

struct unit {                   /* general processing unit */
  int    uid;                   /* integer uniquely identifying each unit */
  char   *label;
  double output;                /* activation level */
  double (*unit_out_f)();       /* note output fcn == activation fcn*/
  double delta;                 /* delta for unit */
  double (*unit_delta_f)();     /* ptr to function to calc delta */
  struct link *inlinks;         /* for propagation */
  struct link *outlinks;        /* for back propagation */
} *pu[TOTAL+1];                 /* one extra for the bias unit */

struct link {                   /* link between two processing units */
  char   *label;
  double weight;                /* connection or link weight */
  double data;                  /* used to hold the change in weights */
  int    from_unit;             /* uid of from unit */
  int    to_unit;               /* uid of to unit */
  struct link *next_inlink;
  struct link *next_outlink;
};

int     iterations = DEFAULT_ITER;
double  learning_rate = 0.2;
double  momentum = 0.9;
double  pattern_err[PATTERNS];

/*
 * Input Patterns
 * {Big Ears, Big Eyes, Big Teeth, Kindly, Wrinkled, Handsome}
 *   unit 0    unit 1     unit 2   unit 3   unit 4    unit 5
 */
double  input_pat[PATTERNS+1][NUM_IN] = {
  {1.0, 1.0, 1.0, 0.0, 0.0, 0.0},       /* Wolf */
  {0.0, 1.0, 0.0, 1.0, 1.0, 0.0},       /* Grandma */
  {1.0, 0.0, 0.0, 1.0, 0.0, 1.0},       /* Woodcutter */
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},       /* Used for Recognize Mode */
};

/*
 * Target Patterns
 * {Scream, Run Away, Look for Woodcutter, Approach, Kiss on Cheek,
 *      Offer Food, Flirt with}
 */
double  target_pat[PATTERNS][NUM_OUT] = {
  {0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1},  /* response to Wolf */
  {0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1},  /* response to Grandma */
  {0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9},  /* response to Woodcutter */
};

/*
 * function declarations
 */
void    print_header();
char    get_command();
double  out_f(), delta_f_out(), delta_f_hid(), random(), pattern_error();


main()
{
  char   ch;
  extern struct unit *pu[];

  print_header();
  create_processing_units(pu);
  create_in_out_links(pu);
  for (;;) {
    ch = get_command("\nEnter Command (Learn, Recognize, Quit) => ");
    switch (ch) {
    case 'l':
    case 'L':
      printf("\n\tLEARN MODE\n\n");
      learn(pu);
      break;
    case 'r':
    case 'R':
      printf("\n\tRECOGNIZE MODE\n\n");
      recognize(pu);
      break;
    case 'q':
    case 'Q':
      exit(1);
      break;
    default:
      fprintf(stderr, "Invalid Command\n");
      break;
    }
  }
}


void
print_header()
{
  printf("%s%s%s",
         "\n\tBPSIM -- Back Propagation Learning Rule Neural Net Simulator\n",
         "\t\t for the tabula rasa Little Red Riding Hood example.\n\n",
         "\t\t Written by Josiah C. Hoskins\n");
}


/*
 * create input, hidden, output units (and threshold or bias unit)
 */
create_processing_units(pu)
struct  unit *pu[];
{
  int   id;                     /* processing unit index */
  struct unit *create_unit();

  for (id = IN_UID(0); id < IN_UID(NUM_IN); id++)
    pu[id] = create_unit(id, "input", 0.0, NULL, 0.0, NULL);
  for (id = HID_UID(0); id < HID_UID(NUM_HID); id++)
    pu[id] = create_unit(id, "hidden", 0.0, out_f, 0.0, delta_f_hid);
  for (id = OUT_UID(0); id < OUT_UID(NUM_OUT); id++)
    pu[id] = create_unit(id, "output", 0.0, out_f, 0.0, delta_f_out);
  pu[BIAS_UID] = create_unit(BIAS_UID, "bias", 1.0, NULL, 0.0, NULL);
}


/*
 * create links - fully connected for each layer
 *                note: the bias unit has one link to ea hid and out unit
 */
create_in_out_links(pu)
struct  unit *pu[];
{
  int   i, j;           /* i == to and j == from unit id's */
  struct link *create_link();

  /* fully connected units */
  for (i = HID_UID(0); i < HID_UID(NUM_HID); i++) { /* links to hidden */
    pu[BIAS_UID]->outlinks =
      pu[i]->inlinks = create_link(pu[i]->inlinks, i,
                                   pu[BIAS_UID]->outlinks, BIAS_UID,
                                   (char *)NULL,
                                   random(), 0.0);
    for (j = IN_UID(0); j < IN_UID(NUM_IN); j++) /* from input units */
      pu[j]->outlinks =
        pu[i]->inlinks = create_link(pu[i]->inlinks, i, pu[j]->outlinks, j,
                                     (char *)NULL, random(), 0.0);
  }
  for (i = OUT_UID(0); i < OUT_UID(NUM_OUT); i++) {     /* links to output */
    pu[BIAS_UID]->outlinks =
            pu[i]->inlinks = create_link(pu[i]->inlinks, i,
                                         pu[BIAS_UID]->outlinks, BIAS_UID,
                                         (char *)NULL, random(), 0.0);
    for (j = HID_UID(0); j < HID_UID(NUM_HID); j++) /* from hidden units */
      pu[j]->outlinks =
        pu[i]->inlinks = create_link(pu[i]->inlinks, i, pu[j]->outlinks, j,
                                     (char *)NULL, random(), 0.0);
  }
}


/*
 * return a random number bet 0.0 and 1.0
 */
double
random()
{
  return((rand() % 32727) / 32737.0);
}


/*
 * the next two functions are general utility functions to create units
 * and create links
 */
struct unit *
create_unit(uid, label, output, out_f, delta, delta_f)
int  uid;
char *label;
double   output, delta;
double   (*out_f)(), (*delta_f)();
{
  struct unit  *unitptr;

/*
  if (!(unitptr = (struct unit *)malloc(sizeof(struct unit)))) {
TURBO C doesnt automatically set pointers to NULL - so use calloc(...,1) */
  if (!(unitptr = (struct unit *)calloc(sizeof(struct unit),1))) {
    fprintf(stderr, "create_unit: not enough memory\n");
    exit(1);
  }
  /* initialize unit data */
  unitptr->uid = uid;
  unitptr->label = label;
  unitptr->output = output;
  unitptr->unit_out_f = out_f;  /* ptr to output fcn */
  unitptr->delta = delta;
  unitptr->unit_delta_f = delta_f;
  return (unitptr);
}


struct link *
create_link(start_inlist, to_uid, start_outlist, from_uid, label, wt, data)
struct  link *start_inlist, *start_outlist;
int     to_uid, from_uid;
char *  label;
double  wt, data;
{
  struct link  *linkptr;

/*  if (!(linkptr = (struct link *)malloc(sizeof(struct link)))) { */
  if (!(linkptr = (struct link *)calloc(sizeof(struct link),1))) {
    fprintf(stderr, "create_link: not enough memory\n");
    exit(1);
  }
  /* initialize link data */
  linkptr->label = label;
  linkptr->from_unit = from_uid;
  linkptr->to_unit = to_uid;
  linkptr->weight = wt;
  linkptr->data = data;
  linkptr->next_inlink = start_inlist;
  linkptr->next_outlink = start_outlist;
  return(linkptr);
}


char
get_command(s)
char    *s;
{
  char  command[BUFSIZ];

  fputs(s, stdout);
  fflush(stdin); fflush(stdout);
  (void)fgets(command, BUFSIZ, stdin);
  return((command[0]));         /* return 1st letter of command */
}


learn(pu)
struct unit *pu[];
{
  register i, temp;
  char   tempstr[BUFSIZ];
  extern int    iterations;
  extern double learning_rate, momentum;
  static char prompt[] = "Enter # iterations (default is 250) => ";
  static char quote1[] = "Perhaps, Little Red Riding Hood ";
  static char quote2[] = "should do more learning.\n";

  printf(prompt);
  fflush(stdin); fflush(stdout);
  gets(tempstr);
  if (temp = atoi(tempstr))
    iterations = temp;

  printf("\nLearning ");
  for (i = 0; i < iterations; i++) {
    if ((i % NOTIFY) == 0) {
      printf(".");
      fflush(stdout);
    }
    bp_learn(pu, (i == iterations-2 || i == iterations-1 || i == iterations));
  }
  printf(" Done\n\n");
  printf("Error for Wolf pattern = \t%lf\n", pattern_err[0]);
  printf("Error for Grandma pattern = \t%lf\n", pattern_err[1]);
  printf("Error for Woodcutter pattern = \t%lf\n", pattern_err[2]);
  if (pattern_err[WOLF_PATTERN] > ERROR_TOLERANCE) {
    printf("\nI don't know the Wolf very well.\n%s%s", quote1, quote2);
  } else if (pattern_err[GRANDMA_PATTERN] > ERROR_TOLERANCE) {
    printf("\nI don't know Grandma very well.\n%s%s", quote1, quote2);
  } else if (pattern_err[WOODCUT_PATTERN] > ERROR_TOLERANCE) {
    printf("\nI don't know Mr. Woodcutter very well.\n%s%s", quote1, quote2);
  } else {
    printf("\nI feel pretty smart, now.\n");
  }
}


/*
 * back propagation learning
 */
bp_learn(pu, save_error)
struct unit *pu[];
int    save_error;
{
  static int count = 0;
  static int pattern = 0;
  extern double pattern_err[PATTERNS];

  init_input_units(pu, pattern); /* initialize input pattern to learn */
  propagate(pu);                 /* calc outputs to check versus targets */
  if (save_error)
    pattern_err[pattern] = pattern_error(pattern, pu);
  bp_adjust_weights(pattern, pu);
  if (pattern < PATTERNS - 1)
    pattern++;
  else
      pattern = 0;
  count++;
}


/*
 * initialize the input units with a specific input pattern to learn
 */
init_input_units(pu, pattern)
struct unit *pu[];
int    pattern;
{
  int   id;

  for (id = IN_UID(0); id < IN_UID(NUM_IN); id++)
    pu[id]->output = input_pat[pattern][id];
}


/*
 * calculate the activation level of each unit
 */
propagate(pu)
struct unit *pu[];
{
  int   id;

  for (id = HID_UID(0); id < HID_UID(NUM_HID); id++)
    (*(pu[id]->unit_out_f))(pu[id], pu);
  for (id = OUT_UID(0); id < OUT_UID(NUM_OUT); id++)
    (*(pu[id]->unit_out_f))(pu[id], pu);
}


/*
 * function to calculate the activation or output of units
 */
double
out_f(pu_ptr, pu)
struct unit *pu_ptr, *pu[];
{
  double sum = 0.0 , exp();
  struct link *tmp_ptr;

  tmp_ptr = pu_ptr->inlinks;
  while (tmp_ptr) {
    /* sum up (outputs from inlinks times weights on the inlinks) */
    sum += pu[tmp_ptr->from_unit]->output * tmp_ptr->weight;
    tmp_ptr = tmp_ptr->next_inlink;
  }
  pu_ptr->output = 1.0/(1.0 + exp(-sum));
}


/*
 * half of the sum of the squares of the errors of the
 * output versus target values
 */
double
pattern_error(pat_num, pu)
int     pat_num;        /* pattern number */
struct  unit *pu[];
{
  int           i;
  double        temp, sum = 0.0;

  for (i = OUT_UID(0); i < OUT_UID(NUM_OUT); i++) {
    temp = target_pat[pat_num][TARGET_INDEX(i)] - pu[i]->output;
    sum += temp * temp;
  }
  return (sum/2.0);
}


bp_adjust_weights(pat_num, pu)
int     pat_num;        /* pattern number */
struct  unit *pu[];
{
  int           i;              /* processing units id */
  double        temp1, temp2, delta, error_sum;
  struct link   *inlink_ptr, *outlink_ptr;

  /* calc deltas */
  for (i = OUT_UID(0); i < OUT_UID(NUM_OUT); i++) /* for each output unit */
    (*(pu[i]->unit_delta_f))(pu, i, pat_num); /* calc delta */
  for (i = HID_UID(0); i < HID_UID(NUM_HID); i++) /* for each hidden unit */
    (*(pu[i]->unit_delta_f))(pu, i);      /* calc delta */
  /* calculate weights */
  for (i = OUT_UID(0); i < OUT_UID(NUM_OUT); i++) {     /* for output units */
    inlink_ptr = pu[i]->inlinks;
    while (inlink_ptr) {        /* for each inlink to output unit */
      temp1 = learning_rate * pu[i]->delta *
        pu[inlink_ptr->from_unit]->output;
      temp2 = momentum * inlink_ptr->data;
      inlink_ptr->data = temp1 + temp2; /* new delta weight */
      inlink_ptr->weight += inlink_ptr->data;   /* new weight */
      inlink_ptr = inlink_ptr->next_inlink;
    }
  }
  for (i = HID_UID(0); i < HID_UID(NUM_HID); i++) { /* for ea hid unit */
    inlink_ptr = pu[i]->inlinks;
    while (inlink_ptr) {        /* for each inlink to output unit */
      temp1 = learning_rate * pu[i]->delta *
        pu[inlink_ptr->from_unit]->output;
      temp2 = momentum * inlink_ptr->data;
      inlink_ptr->data = temp1 + temp2; /* new delta weight */
      inlink_ptr->weight += inlink_ptr->data;   /* new weight */
        inlink_ptr = inlink_ptr->next_inlink;
    }
  }
}


/*
 * calculate the delta for an output unit
 */
double
delta_f_out(pu, uid, pat_num)
struct unit *pu[];
int    uid, pat_num;
{
  double        temp1, temp2, delta;

  /* calc deltas */
  temp1 = (target_pat[pat_num][TARGET_INDEX(uid)] - pu[uid]->output);
  temp2 = (1.0 - pu[uid]->output);
  delta = temp1 * pu[uid]->output * temp2; /* calc delta */
  pu[uid]->delta = delta; /* store delta to pass on */
}


/*
 * calculate the delta for a hidden unit
 */
double
delta_f_hid(pu, uid)
struct unit *pu[];
int    uid;
{
  double        temp1, temp2, delta, error_sum;
  struct link   *inlink_ptr, *outlink_ptr;

  outlink_ptr = pu[uid]->outlinks;
  error_sum = 0.0;
  while (outlink_ptr) {
    error_sum += pu[outlink_ptr->to_unit]->delta * outlink_ptr->weight;
    outlink_ptr = outlink_ptr->next_outlink;
  }
  delta = pu[uid]->output * (1.0 - pu[uid]->output) * error_sum;
  pu[uid]->delta = delta;
}


recognize(pu)
struct unit *pu[];
{
  int    i;
  char   tempstr[BUFSIZ];
  static char *p[] = {"Big Ears?", "Big Eyes?", "Big Teeth?",
                      "Kindly?\t", "Wrinkled?", "Handsome?"};

  for (i = 0; i < NUM_IN; i++) {
    printf("%s\t(y/n) ", p[i]);
    fflush(stdin); fflush(stdout);
    fgets(tempstr, BUFSIZ, stdin);
    if (tempstr[0] == 'Y' || tempstr[0] == 'y')
      input_pat[PATTERNS][i] = 1.0;
    else
      input_pat[PATTERNS][i] = 0.0;
  }
  init_input_units(pu, PATTERNS);
  propagate(pu);
  print_behaviour(pu);
}


print_behaviour(pu)
struct unit *pu[];
{
  int   id, count = 0;
  static char *behaviour[] = {
    "Screams", "Runs Away", "Looks for Woodcutter", "Approaches",
    "Kisses on Cheek", "Offers Food", "Flirts with Woodcutter" };

  printf("\nLittle Red Riding Hood: \n");
  for (id = OUT_UID(0); id < OUT_UID(NUM_OUT); id++){ /* links to out units */
    if (pu[id]->output > ON_TOLERANCE)
      printf("\t%s\n", behaviour[count]);
    count++;
  }
  printf("\n");
}

/*
! Thomas Muhr    Knowledge-Based Systems Dept. Technical University of Berlin !
! BITNET/EARN:   muhrth@db0tui11.bitnet                                       !
! UUCP:          morus@netmbx.UUCP (Please don't use from outside Germany)    !
! BTX:           030874162  Tel.: (Germany 0049) (Berlin 030) 87 41 62        !
*/
