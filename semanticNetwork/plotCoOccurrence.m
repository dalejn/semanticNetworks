data = load('coOccurrenceMatrix.txt');
    
%%

G = graph(data);
G.Nodes.Name = {'meno' 'tell' 'socrates' 'whether' 'virtue' 'acquired' 'teaching' 'practice' 'neither' 'comes' 'nature' 'time' 'thessalians' 'famous' 'hellenes' 'riches' 'riding' 'mistaken' 'equally' 'wisdom' 'especially' 'larisa' 'native' 'city' 'friend' 'aristippus' 'gorgias' 'came' 'flower' 'aleuadae' 'admirer' 'chiefs' 'fell' 'love' 'taught' 'habit' 'answering' 'questions' 'grand' 'bold' 'style' 'becomes' 'know' 'answers' 'comers' 'hellene' 'likes' 'anything' 'different' 'dear' 'athens' 'dearth' 'commodity' 'seems' 'emigrated' 'certain' 'athenian' 'natural' 'would' 'laugh' 'face' 'stranger' 'good' 'opinion' 'think' 'answer' 'question' 'literally' 'much' 'less' 'living' 'region' 'poverty' 'poor' 'rest' 'world' 'confess' 'shame' 'nothing' 'quid' 'quale' 'knew' 'could' 'fair' 'opposite' 'rich' 'noble' 'reverse' 'indeed' 'earnest' 'saying' 'carry' 'back' 'report' 'thessaly' 'never' 'known' 'else' 'judgment' 'memory' 'therefore' 'thought' 'dare' 'said' 'please' 'remind' 'rather' 'view' 'suspect' 'alike' 'true' 'mind' 'gods' 'generous' 'shall' 'truly' 'delighted' 'find' 'really' 'knowledge' 'although' 'found' 'anybody' 'difficulty' 'take' 'first' 'administer' 'state' 'administration' 'benefit' 'friends' 'harm' 'enemies' 'must' 'careful' 'suffer' 'woman' 'wish' 'easily' 'described' 'duty' 'order' 'house' 'keep' 'indoors' 'obey' 'husband' 'every' 'condition' 'life' 'young' 'male' 'female' 'bond' 'free' 'virtues' 'numberless' 'lack' 'definitions' 'relative' 'actions' 'ages' 'vice' 'compare' 'arist' 'fortunate' 'present' 'swarm' 'theaet' 'keeping' 'suppose' 'many' 'kinds' 'bees' 'reply' 'differ' 'distinguished' 'quality' 'example' 'beauty' 'size' 'shape' 'another' 'went' 'desire' 'able' 'common' 'makes' 'well' 'fixed' 'understand' 'beginning' 'hold' 'child' 'apply' 'health' 'strength' 'always' 'strong' 'reason' 'form' 'subsisting' 'mean' 'difference' 'grown-up' 'person' 'help' 'feeling' 'case' 'others' 'either' 'ordered' 'without' 'temperance' 'justice' 'certainly' 'temperately' 'justly' 'women' 'elder' 'intemperate' 'unjust' 'temperate' 'participation' 'inference' 'surely' 'unless' 'sameness' 'proven' 'remember' 'definition' 'seeking' 'want' 'power' 'governing' 'mankind' 'include' 'slave' 'govern' 'father' 'master' 'governed' 'longer' 'small' 'according' 'unjustly' 'agree' 'might' 'round' 'simply' 'adopt' 'mode' 'speaking' 'figures' 'quite' 'right' 'names' 'asked' 'courage' 'magnanimity' 'searching' 'though' 'unable' 'runs' 'even' 'follow' 'attempt' 'notion' 'things' 'wonder' 'nearer' 'answered' 'roundness' 'asking' 'proceeded' 'told' 'similarly' 'colour' 'whiteness' 'questioner' 'rejoined' 'colours' 'pursue' 'matter' 'ever' 'anon' 'landed' 'particulars' 'since' 'call' 'name' 'opposed' 'designate' 'contains' 'straight' 'assert' 'give' 'look' 'astonished' 'looking' 'simile' 'multis' 'includes' 'indulge' 'means' 'best' 'prize' 'explain' 'thing' 'follows' 'satisfied' 'sure' 'similar' 'simple' 'granted' 'sort' 'given' 'truth' 'philosopher' 'eristic' 'antagonistic' 'wrong' 'business' 'argument' 'refute' 'talking' 'milder' 'strain' 'dialecticians' 'vein' 'speak' 'make' 'premises' 'interrogated' 'willing' 'admit' 'endeavour' 'approach' 'acknowledge' 'termination' 'extremity' 'words' 'sense' 'aware' 'prodicus' 'draw' 'distinctions' 'still' 'ended' 'terminated' 'difficult' 'believe' 'meaning' 'surface' 'solid' 'geometry' 'define' 'ends' 'concisely' 'limit' 'outrageous' 'plaguing' 'trouble' 'remembering' 'blindfolded' 'hear' 'creature' 'lovers' 'imperatives' 'like' 'beauties' 'prime' 'tyrannical' 'weakness' 'humour' 'manner' 'familiar' 'better' 'empedocles' 'effluences' 'existence' 'passages' 'pass' 'exactly' 'large' 'sight' 'pindar' 'says' 'read' 'effluence' 'commensurate' 'palpable' 'appears' 'admirable' 'happens' 'hearing' 'discovered' 'sound' 'smell' 'phenomena' 'orthodox' 'solemn' 'acceptable' 'alexidemus' 'thinking' 'stay' 'initiated' 'compelled' 'yesterday' 'away' 'mysteries' 'sake' 'afraid' 'turn' 'fulfil' 'promise' 'universal' 'singular' 'plural' 'facetious' 'break' 'deliver' 'whole' 'broken' 'number' 'pieces' 'pattern' 'desires' 'honourable' 'provide' 'poet' 'attaining' 'evil' 'evils' 'imagine' 'knows' 'notwithstanding' 'possession' 'possesses' 'obvious' 'ignorant' 'goods' 'hurtful' 'possessor' 'hurt' 'miserable' 'proportion' 'inflicted' 'upon' 'otherwise' 'ill-fated' 'misery' 'nobody' 'affirmed' 'respect' 'desiring' 'appear' 'entirely' 'approve' 'point' 'likely' 'affirm' 'wealth' 'gold' 'silver' 'office' 'honour' 'hereditary' 'great' 'king' 'getting' 'gained' 'piously' 'deem' 'consequence' 'acquisition' 'dishonest' 'deemed' 'holiness' 'part' 'accompany' 'mere' 'non-acquisition' 'oneself' 'whatever' 'accompanied' 'honesty' 'devoid' 'mock' 'hands' 'unbroken' 'gave' 'frame' 'forgotten' 'already' 'admissions' 'parts' 'telling' 'declare' 'action' 'done' 'frittered' 'little' 'fear' 'begin' 'repeat' 'ought' 'rejected' 'terms' 'unexplained' 'unadmitted' 'portion' 'fashion' 'used' 'doubting' 'making' 'doubt' 'casting' 'spells' 'bewitched' 'enchanted' 'wits' 'venture' 'jest' 'seem' 'appearance' 'flat' 'torpedo' 'fish' 'torpifies' 'come' 'near' 'touch' 'torpified' 'soul' 'tongue' 'torpid' 'delivered' 'infinite' 'variety' 'speeches' 'persons' 'ones' 'moment' 'wise' 'voyaging' 'going' 'home' 'places' 'cast' 'prison' 'magician' 'rogue' 'caught' 'made' 'pretty' 'gentlemen' 'similes' 'return' 'compliment' 'cause' 'torpidity' 'perplex' 'clear' 'utterly' 'perplexed' 'perhaps' 'touched' 'objection' 'join' 'enquiry' 'enquire' 'forth' 'subject' 'tiresome' 'dispute' 'introducing' 'argue' 'need' 'heard' 'spoke' 'divine' 'glorious' 'conceive' 'priests' 'priestesses' 'studied' 'profession' 'poets' 'inspiration' 'inspired' 'mark' 'immortal' 'termed' 'dying' 'born' 'destroyed' 'moral' 'live' 'perfect' 'ninth' 'year' 'persephone' 'sends' 'souls' 'received' 'penalty' 'ancient' 'crime' 'beneath' 'light' 'become' 'kings' 'mighty' 'called' 'saintly' 'heroes' 'times' 'seen' 'exist' 'remembrance' 'everything' 'akin' 'learned' 'eliciting' 'learning' 'single' 'recollection' 'strenuous' 'faint' 'listen' 'sophistical' 'impossibility' 'idle' 'sweet' 'sluggard' 'active' 'inquisitive' 'confiding' 'gladly' 'learn' 'process' 'teach' 'involve' 'contradiction' 'protest' 'intention' 'prove' 'easy' 'utmost' 'numerous' 'attendants' 'demonstrate' 'hither' 'greek' 'speaks' 'attend' 'observe' 'learns' 'remembers' 'square' 'lines' 'equal' 'drawn' 'middle' 'side' 'feet' 'direction' 'space' 'foot' 'taken' 'twice' 'count' 'length' 'line' 'forms' 'double' 'clearly' 'fancies' 'long' 'necessary' 'produce' 'guesses' 'recalls' 'steps' 'regular' 'oblong' 'doubled' 'containing' 'describe' 'divisions' 'sixteen' 'gives' 'half' 'greater' 'evident' 'reckon' 'show' 'advances' 'confidently' 'knowing' 'ignorance' 'torpedos' 'shock' 'assisted' 'degree' 'discovery' 'remedy' 'ready' 'enquired' 'fancied' 'fallen' 'perplexity' 'idea' 'desired' 'farther' 'development' 'share' 'watch' 'explaining' 'instead' 'former' 'third' 'fill' 'vacant' 'corner' 'spaces' 'larger' 'reaching' 'bisect' 'contain' 'interior' 'section' 'extends' 'diagonal' 'proper' 'prepared' 'head' 'notions' 'stirred' 'dream' 'frequently' 'last' 'recover' 'spontaneous' 'recovery' 'possessed' 'branch' 'bred' 'fact' 'undeniable' 'acquire' 'thoughts' 'awakened' 'putting' 'obviously' 'existed' 'wherefore' 'cheer' 'recollect' 'feel' 'somehow' 'altogether' 'confident' 'braver' 'helpless' 'indulged' 'fancy' 'theme' 'fight' 'word' 'deed' 'excellent' 'agreed' 'effort' 'together' 'original' 'regard' 'gift' 'coming' 'command' 'instruction' 'ascertained' 'controlling' 'freedom' 'yield' 'irresistible' 'qualities' 'rate' 'condescend' 'allow' 'argued' 'hypothesis' 'geometrician' 'triangle' 'capable' 'inscribed' 'circle' 'area' 'offer' 'assist' 'forming' 'conclusion' 'produced' 'diameter' 'autou' 'falls' 'short' 'corresponding' 'applied' 'impossible' 'assume' 'geometrical' 'class' 'mental' 'remembered' 'disputing' 'alone' 'quick' 'next' 'species' 'aside' 'distinct' 'embraces' 'profitable' 'severally' 'profit' 'sometimes' 'guiding' 'principle' 'rightly' 'consider' 'quickness' 'apprehension' 'wanting' 'prudence' 'confidence' 'harmed' 'profited' 'general' 'attempts' 'endures' 'guidance' 'happiness' 'folly' 'admitted' 'none' 'addition' 'accordingly' 'guides' 'uses' 'wrongly' 'benefited' 'foolish' 'universally' 'human' 'hang' 'inferred' 'profits' 'arrive' 'wholly' 'partly' 'assuredly' 'discerners' 'characters' 'future' 'showing' 'adopted' 'kept' 'citadel' 'stamp' 'piece' 'tamper' 'grew' 'useful' 'alternative' 'supposition' 'erroneous' 'soundness' 'stand' 'firm' 'slow' 'heart' 'retract' 'assertion' 'teachers' 'disciples' 'conversely' 'assumed' 'incapable' 'often' 'pains' 'succeeded' 'search' 'wanted' 'fortunately' 'sitting' 'anytus' 'repair' 'place' 'wealthy' 'anthemion' 'accident' 'ismenias' 'theban' 'recently' 'polycrates' 'skill' 'industry' 'well-conditioned' 'modest' 'insolent' 'overbearing' 'annoying' 'moreover' 'education' 'people' 'choose' 'highest' 'offices' 'physician' 'send' 'physicians' 'cobbler' 'cobblers' 'sending' 'profess' 'demand' 'payment' 'reasons' 'flute-playing' 'arts' 'flute-player' 'refuse' 'money' 'professed' 'disciple' 'wishes' 'conduct' 'height' 'zeus' 'position' 'advise' 'attain' 'kind' 'parents' 'receive' 'citizens' 'strangers' 'previous' 'imply' 'avouch' 'hellas' 'impart' 'price' 'sophists' 'heracles' 'forbear' 'hope' 'kinsman' 'acquaintance' 'mine' 'citizen' 'corrupted' 'manifest' 'pest' 'corrupting' 'influence' 'positively' 'corrupt' 'entrusted' 'disservice' 'protagoras' 'craft' 'illustrious' 'pheidias' 'created' 'works' 'statuaries' 'mender' 'shoes' 'patcher' 'clothes' 'worse' 'remained' 'thirty' 'days' 'undetected' 'soon' 'starved' 'whereas' 'forty' 'years' 'seventy' 'death' 'spent' 'reputation' 'retains' 'spoken' 'lived' 'deceived' 'youth' 'supposed' 'consciously' 'unconsciously' 'wisest' 'minds' 'relations' 'guardians' 'care' 'cities' 'allowed' 'drive' 'wronged' 'angry' 'belongings' 'unacquainted' 'acquainted' 'diviner' 'judging' 'enquiring' 'eminent' 'describing' 'family' 'oblige' 'fault' 'athenians' 'individuals' 'gentleman' 'random' 'grow' 'nevertheless' 'generation' 'statesmen' 'discussing' 'communicated' 'imparted' 'arguing' 'themistocles' 'teacher' 'jealous' 'intentionally' 'abstained' 'imparting' 'cleophantus' 'horseman' 'upright' 'horseback' 'hurl' 'javelin' 'marvellous' 'trained' 'elders' 'showed' 'capacity' 'sought' 'train' 'minor' 'accomplishments' 'neighbours' 'excelled' 'past' 'aristides' 'lysimachus' 'masters' 'result' 'mortal' 'pericles' 'magnificent' 'sons' 'paralus' 'xanthippus' 'unrivalled' 'horsemen' 'music' 'gymnastics' 'sorts' 'respects' 'level' 'wished' 'incompetent' 'meaner' 'thucydides' 'melesias' 'stephanus' 'besides' 'giving' 'wrestling' 'wrestlers' 'committed' 'xanthias' 'eudorus' 'celebrated' 'whose' 'children' 'spend' 'cost' 'allies' 'foreigner' 'spare' 'cares' 'advice' 'recommend' 'easier' 'rage' 'thinks' 'defaming' 'second' 'defamation' 'forgive' 'meanwhile' 'possibility' 'vocation' 'professors' 'promising' 'hears' 'laughs' 'politicians' 'doubts' 'theognis' 'elegiac' 'verses' 'theog' 'drink' 'agreeable' 'lose' 'intelligence' 'shifts' 'understanding' 'perform' 'feat' 'obtained' 'rewards' 'sprung' 'sire' 'voice' 'remark' 'professing' 'acknowledged' 'ideas' 'confusion' 'anywhere' 'scholars' 'educator' 'improve' 'discussion' 'remarked' 'possible' 'episteme' 'denied' 'seeing' 'necessarily' 'admitting' 'supposing' 'guide' 'phrhonesis' 'thither' 'correct' 'omitted' 'speculation' 'cogency' 'preferred' 'observed' 'images' 'daedalus' 'euthyphro' 'country' 'require' 'fastened' 'play' 'truant' 'valuable' 'possessions' 'liberty' 'walk' 'runaway' 'slaves' 'value' 'beautiful' 'illustration' 'opinions' 'abide' 'fruitful' 'remain' 'fastening' 'bound' 'abiding' 'chain' 'conjecture' 'differs' 'leading' 'perfects' 'whit' 'inferior' 'states' 'excluded' 'happen' 'chance' 'political' 'grounded' 'probably' 'remains' 'guided' 'politics' 'divination' 'religion' 'diviners' 'prophets' 'succeed' 'calling' 'including' 'tribe' 'illumined' 'spartans' 'praise' 'offence' 'opportunity' 'instinct' 'virtuous' 'educating' 'homer' 'tiresias' 'dead' 'flitting' 'shades' 'reality' 'shadows' 'actual' 'persuaded' 'persuade' 'exasperated' 'conciliate' 'service'}';

plot(G,'Layout','force', 'NodeLabel',G.Nodes.Name)