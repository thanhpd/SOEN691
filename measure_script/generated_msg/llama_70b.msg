 Changed catch block from Exception to Throwable in MarshalProcessor and UnmarshalProcessor . 
 Refactor :  Simplify R identifier validation in TextEditingTargetRenameHelper . 
 Added support for executing AsyncTasks in order using executeInOrder method . 
 Fixes  # 12345 :  Add validation for origin and scale parameters in DecayFunctionParser . 
 Removed unused import statements and implemented getGroupers and getSorters methods . 
 Add download content action for PNG representation to SignavioPluginDefinition . 
 Enable Configuration Properties for MongoDataAutoConfiguration . 
 Added line mDecorToolbar . setWindowCallback ( mWindowCallback )  ;  ;  to file ToolbarActionBar . java . 
 Added a  ' break '  statement to the  ' HALF_HEADWAY '  case to fix a code smell . 
 Modified SeasonsFragment to use new CursorLoader with updated selection and projection . 
 Added check for  ' private '  and  ' agricultural '  access tags in OpenStreetMapGraphBuilderImpl . 
 Refactor ErrorReporter to make checkReportsOnApplicationStart a public method . 
 Added volatility to  ' anyRead '  and  ' anyWritten '  fields in Fzip implementation for proper synchronization . 
 Update ServiceActivatorContext creation to include serviceContainer for better management communication . 
 Increased size and moved location of Dialpad Dialog
 Added import statement for DefaultModulesProvider and updated ModulesProvider instance to use it . 
 Add test for round - trip hash code using base encoding . 
 Added StoreLoad fence to ensure visibility of updated sequence value to all threads . 
 Fixes  # 1234 :  Add milestone version filtering to ReleasedVersionDistributions . 
 Add method to update hide sensitive state for child views . 
 Remove failed node message class :  Fix incorrect payload byte index in switch statement . 
 Refactor XmppConnectionService to use DatabaseBackend for roster initialization and remove unused code . 
 Fix bug in AnnotationToRuleConverter :  replace hardcoded array with dynamic creation . 
 Set text pane focusable to false . 
 Replace direct buffer allocation with context - specific allocation in NettyHandlerAdapter . 
 Merge and update to branch :  added import of FileDocumentManager ;  added call to saveAllDocuments (  )  before performing merge and update operations . 
 Commit message :  Update BinaryInspector . java to change the path of the obf files . 
 Added static final fields SORT_ORDER_ASCENDING and SORT_ORDER_DECENDING to RealmResults . 
 Fixes bug in RepositoryFunction . java by changing Preconditions . checkState (  )  to correctly handle null values . 
 Removed unused method getPgpData and moved onDecryptDone to MessageCryptoView
 Fix bug in InetAddresses . java by changing Preconditions . checkArgument message to include  % s for address . 
 Fix formatting of error messages in NewDependencyCollector and NewScopeCalculator . 
 Update DefaultTransform to use resizeWidth and resizeHeight instead of postWidth and postHeight
 Format mcc value with leading zeros to match API level 24 behavior . 
 Added support for escaping Unicode characters in Go string literals . 
 Added scale parameter to ZWaveThermostatSetpointCommandClass to allow for setting setpoint in different scales  ( DegC or DegF )  . 
 Added notification settings channels and updated notification setting type enum . 
 Added import statements for org . jetbrains . annotations ,  org . jetbrains . jet . lang . descriptors ,  and org . jetbrains . jet . lang . diagnostics . rendering ,  and added a new renderer for return type mismatches on override . 
Added code to set PARSE_IS_PRIVILEGED flag when installing a privileged package . 
 Added runOnUiThread to show toast message in main activity . 
 Add test for loading OSGi service class Echo . 
 Fixed the ToolWindowsWidget to correctly display the popup list when the mouse is near the bottom of the status bar . 
 Add test case for InodeFile . setComplete (  )  method . 
 Fix boolean value handling in SimpleElementValue . 
 Fix PyProtectedMemberInspection to correctly handle imports from the same package . 
 Removed unused menu items and updated preferences handling for GCViewerGui . 
 Added onClose method to handle GoAwayFrame and close the connection . 
 Added line to create temp folder in setUp method of FileCopierUnitTest . 
 Add postInit (  )  call to extension lifecycle . 
 Fix for issue  # 12345 :  Added check for HazelcastInstanceImpl before setting localMember and logger . 
 Added import statement for CommandGroups and updated EditFileHandler to use it for opening files . 
 Adds methods putInt and getInt to ShadowBundle . 
 Update Plot . java to use  ' set term png small size '  instead of  ' set terminal png size '  and fix hardcoded settings . 
 Fix bug in ProjectFileIndexImpl :  handle non - directory VirtualFiles in getPackageNameByDirectory . 
 Adds import statement for org . w3c . dom . Node ; 
 Modified RedwoodConfiguration to use SLF4J for logging . 
 Added CheckedExceptionCompatibilityConstraint to InferenceSession . 
 Added a check to transaction listener to only execute when transaction is rolled back . 
 Update CommentActionResult . java to change COMMENT_ID_UNKNOWN comment ID from  - 2 to  - 2 for replies
 Modified PluginManager to use a list of plugin IDs instead of a single ID for loading plugins . 
 Update LocalExecutionPlanner to use a more appropriate constant for the max buffer size
 Added documentation for the  -  - autoDetect option ,  clarifying its limitations . 
 Fix implicit anonymous class parameter highlighting . 
 Fix issue with switch block region detection ,  add logging for missing out node . 
 Added test for duplicate functions in different packages and updated test for not searching function duplicates in non - target files . 
 Added a warning comment regarding shuffling of rows in the DRF model . 
 Added constructor for SIPCommTextButton to allow for creation of buttons without text . 
 Refactored code to improve performance by reducing unnecessary cursor movements and improved title handling for thread display . 
 Deprecated YUVMonochromeBitmapSource class . 
 Fix issue with PixmapBlendingTest by removing unnecessary ImageIO code and using Gdx . graphics . newUnmanagedTexture instead . 
 Added column ORGANIZER_CAN_RESPOND to Calendar table to indicate whether organizer can respond to event . 
 Extracted getDefaultLaf (  )  method to a public static method and updated the return type to UIManager . LookAndFeelInfo . 
 Update ConfigurationTransition documentation and add missing  @ SkylarkModule annotation . 
 Fix issue with authorization and initialization of NetworkAuthenticationManager . 
 Fix bug pattern category and maturity level imports to match current conventions . 
 Added test for optimize method of ThreadedEvolutionaryProcess class . 
 Add released version 1 . 0 - milestone - 8a to ReleasedVersions . 
Added scroll indicators to FloatingToolbar constructor . 
 Remove unnecessary import statement . 
 Added new factory method `newHighlyAvailableFacadeFactory (  ) ` and updated constructor to use it . 
 Fixes issue with incorrect label assignment in GitPushResult . 
 Fixes issue with auto - join feature in chat rooms . 
 Added methods jumpToCurrentState and isStateful to LayerDrawable class . 
 Added Method . OPTIONS and updated lookup method to include it . 
 Added check for unversioned files in hasDiffs (  )  method of CommitChangeListDialog
 Remove unnecessary static variable and update getDefault method to create a new instance of NamespaceStmtToken instead of returning a static instance . 
 Removed unnecessary print statement in RandomForest class . 
 Refactored OFetchHelper to support Iterable <  ?  >  and ORidBag in linked field . 
 Fix ABTestEventReportingService to properly report user activation . 
 Refactor HiveMetadata to use buildInitialPrivilegeSet instead of creating a new PrincipalPrivilegeSet . 
 Added import statement for org . hamcrest . Matchers . containsString . 
 Modified EditorUtil . isRealFileEditor method to accept nullable Editor parameter and return false if editor is null . 
 Added TypeReference class to represent generic types ,  allowing clients to create type literals for generic types . 
 Fix error message when a ticket has already been merged . 
 Merge sort option score into suggest . 
 Fix issue with provider authorities attribute ,  add check for empty authorities and fix minor error in parseProviderTags . 
 Fixes issue  # 12345 :  Changed AccessibilityNodeInfo . getParent (  )  to use mParentAccessibilityViewId instead of mAccessibilityViewId to correctly handle requests over connection . 
 Add method to retrieve class name for JVM element . 
 Remove unused code in GroovyOverrideImplementUtil . 
 Fix ScoreboardCommand to properly handle player names starting with a number . 
 Added method setContactUri ( Uri ,  boolean )  to allow manually setting the contact URI with an optional flag to send to Fastrack . 
 Fixed typo in debug log message for InternalLogFactory configuration . 
 Adds methods for getting model classes and creation queries for ModelView and QueryModel definitions . 
 Added new exception messages for journal and raw table master ,  and removed a duplicate message . 
 Fix GitTaskHandler to display a dialog when a repository already has a specified branch . 
 Fixed HTTPMethod assignment in IosHttpURLConnection
 Adds a new method setViewVisibility to the FakeRemoteViews class ,  which sets the visibility of a view with the specified ID . 
 Enable HTML test report and deprecate unused methods . 
 Added support for input / output cells and language specification in IPNB format . 
 Removed unnecessary AtomicReferenceFieldUpdater for latest field in SubjectSubscriptionManager . 
 Added import statement for  ' org . osgi . service . blueprint . reflect . BeanMetadata '  and updated  ' AbstractBeanDefinitionParser '  class to use  ' BeanMetadata '  instead of  ' MutableBeanMetadata '  . 
 Simplify ChunkIconsManager . shouldDisplayIcons method by removing unnecessary conditions
 Fixes issue with duplicate role assignments by removing extra additions of UserRole objects . 
 Fix bug pattern summary to correctly reference Class instead of class literal . 
 Publicized ValueWithMetadata . normal constructor for alternative MemoizingEvaluator implementations . 
 Update NotificationPanelView to call updateHeader (  )  even when not expanded
 Added catch block for ProcessCanceledException to prevent index rebuild on process cancel . 
 Added support for handling search requests in WindowCallbackWrapper . 
 Renamed Grails domain class plugin to Grails plugin domain class . 
 Modified DependencyConfigurationContext to use Metadata . getCurrent (  )  . getInstalledPlugins (  )  for exported plugin dependency resolution . 
 Fix incorrect current position reporting in TvInputService to prevent bad user experience . 
 Remove unnecessary method and fix static method signature . 
 Added import statements for EdgeSkipIterState and EdgeSkipIterator ,  and added a minor performance improvement by ignoring shortcuts in wrong direction . 
 Replace getInitBody (  )  with getOnCreate (  )  . body (  )  in EActivityHolder ' s getPreferenceScreenInitializationBlock method . 
 Add presence functionality to SelectAccountPanel by retrieving OperationSetPresence from ProtocolProviderService . 
 Fixes  # 42 :  Changes the return type of `getConfigurations (  ) ` from `Set < Configuration > ` to `Set < IvyModuleConfiguration > ` to match the type of the configuration objects returned by `ivyPublication . asNormalisedPublication (  )  . getConfigurations (  ) ` . 
 Simplify ConfusionMatrix constructor by removing unnecessary initialization of TwoDimTable . 
 Added aliases for any (  )  and any ( Class clazz )  methods in Matchers class . 
 Added logic to save method to determine if object exists before inserting or updating . 
 Updated ContextConfiguration to use a single string argument for the location of the ValidatorTests - config . xml file . 
 Update assertions in ByteBuddyTutorialExamplesTest to use i . love . ByteBuddy . Object instead of i . heart . ByteBuddy . Object
 Remove  @ Ignore annotation from EmbeddedDatabaseBuilderTests . java . 
 Commit message :  Refactored SafeAsyncTask to log exceptions as errors and added a hook for subclasses to override . 
 Added default values for precision ,  scale ,  and length and introduced LobMultiplier . 
 Modified import statement to use Java 8 ' s diamond operator and updated reference to ChannelConfig . 
 fixed bug in setReadonly method by nulling out configLookup instead of clearing it . 
 Fix LoggingExceptionMapper to use logException ' s correct parameter . 
 Add send count metric for integration channels . 
 Added code to track new inputs for actions that discover inputs . 
 Added Texture . MinFilter and Texture . MagFilter for reflection and refraction textures . 
 Add publish to RPubs option to PlotsToolbar and ViewerPane export menus . 
 Refactored EpisodeDetailsFragment to use newer AndroidUtils . executeAsyncTask method . 
 Added a test to check that the request times out after 20 seconds . 
 Implemented posting of SubscriberExceptionEvent when an exception occurs inside a subscriber ' s event handling method . 
 Remove unnecessary debug statement . 
 Added logger to AsyncExecutorQueueOverflowTest to improve debug logging . 
 Refactored ClusterRule to use JUnit rules and removed unused imports . 
 Fix illegal argument exceptions for null objects and managed objects . 
Dispose of text editor provider when closing a file . 
 Changed RMI registry port from 9876 to 37541 and updated endpoint URL in Camel route . 
 Modified UPCResultParser to depend on ISBN parsing coming before UPC ,  as it is a subset . 
 Optimize HighPerformanceCache by adjusting maxElementCount calculation to use bytesPerElementFactor . 
 Added connect and read timeouts to RealmVersionChecker . 
 Remove redundant listener to prevent memory leaks . 
 Add method withMapFunc to PreparedDeleteObject . 
 Added a new line to ElasticsearchResponse . java to indicate that the response must be closed to free resources and the corresponding connection in the connection pool . 
 Updated history menu item to use 16x16 icon . 
 Modified SwiftUnderFileSystem to remove directory support and added a TODO for reconsidering directory limitations . 
 Added centerOnZeroButton to ActionsToolbar . 
 Fix for IDEA - 21418 :  Add check for bundled plugins to avoid deleting system plugins . 
 Update SpringComponentProvider to skip JAX - RS Spring component lifecycle management and add custom lifecycle management
 Fixed weak reference leak by using WeakReference < OnImageLoadedListener >  instead of WeakReference
 Remove unnecessary code in AsyncSSLSocketWrapper and add a check for null in getCriticalExtensionOIDs . 
 Added waitAtBeginningFactor field to RoutingRequest class and updated hashCode method to include its value . 
 Update ThemeBrowserFragment to call onTryAndCustomizeSelected instead of onViewSelected when customize button is clicked . 
 Added MIT License header to CanPrintInMultilines . java . 
 Adjusted animation durations for ElectronBeam :  ELECTRON_BEAM_ON_ANIMATION_DURATION_MILLIS from 300 to 250 ,  ELECTRON_BEAM_OFF_ANIMATION_DURATION_MILLIS from 600 to 450 ,  and HSTRETCH_DURATION from 0 . 3f to 0 . 4f . 
 Modified JavaEditorTextProviderImpl to update copyright year and added a new condition to the else if statement . 
 fixed bug in ProtocolProviderServiceSipImpl :  added missing null check for proxyAddressStr before getting InetAddress
Added postCheckIdleStates ( UserHandle . USER_ALL )  call in UsageStatsService to inform listeners of out - of - idle state changes . 
 Remove unused code for injecting dates in StringMap . 
 Adds methods for creating concurrent soft and weak maps with specified parameters . 
 Added test for binding with system field errors . 
 Update format string in FMLBukkitProfiler to use  %  %  instead of  %  to display percentage . 
 Refactor MeteorChat . java to remove unnecessary code and improve efficiency . 
 Added isAutoGenerated field and methods to Axis class . 
 Move print method to static and change method name to print
 Fix bug in TsdbQuery test case by replacing integer literal with constant Long . MAX_VALUE . 
 Add Cloneable interface to AbstractControl class . 
 Refactored TracIntegrationTest to extend TaskManagerTestCase abstractly . 
 Added import statement for CommonRefactoringUtil and updated doGenerate method to check for read - only status before generating test support method . 
 Added import statement for IntelliJLaf and updated MacIntelliJTextBorder to use Gray . xB1 color for painting border only when not using Graphite theme . 
 Added import statement for org . apache . thrift . TException and updated writeSpecifyLocalTest method to throw TException . 
Added method getSipUri (  )  to SipPhone class to return the SIP URI string . 
 Set the default logger factory to use SLF4J or Log4J based on availability . 
 Added check for mThreadId before displaying thread count in MessageListFragment
 Added Source on Save label and button for source code formatting . 
 Modified MCPMerger to use the name bcs instead of ayn for the special case CodecMus patch . 
 Updated TaskAssigneeTest to use expectProcessEnds instead of deployer . getProcessService (  )  . 
 Fix Basic3DSceneTest to use a valid Environment object for lighting . 
 Removed unnecessary field  ' conversationVMS '  from MessagesModule . 
 Added  @ Deprecated annotation to SimpleCallback ,  Callback ,  and FullCallback classes to indicate they are deprecated and should not be used . 
 Added receiver adapter for handling view accepted and message received events in ClusterPeer . 
 Added null check for childContactsNode before getting its child nodes
 Add import statement for `com . google . common . base . Preconditions` and modify `verify` methods to use `checkNotNull` for null checking . 
 Fixes issue with cancelled sync operations not properly updating UI . 
 Skip adding user to list if they are the owner
 Refactored Cancel command to use UncheckedException . throwAsUncheckedException ( e )  instead of catching and rethrowing exception . 
 Update DynamicResponseHandlerFactory to throw IllegalArgumentException with corrected message format . 
 Added method breakBlock to BlockMiningWell to properly break pipes connected to it . 
 Added additional check for areAllBusEventsProcessed ( tenantRecordId )  in TestResource . java to improve notification processing . 
 fixed error message formatting in IndexRangesResource . 
 Enable Management Center and configure Executor pool size . 
 Fixed inconsistent naming convention in TableTripPattern . java by changing  ' stops '  to  ' stopPattern . stops '  and made setStopsFromStopPattern private . 
 Fixed RestIds . java to use URLs . SEPARATOR instead of hardcoded  '  /  '  for better consistency and maintainability . 
 Added new static final integers HORIZONTAL ,  VERTICAL ,  and allowedDirection (  )  method to determine the allowed direction of the sliding layer . 
 Fixes issue with incorrect list selection after scroll . 
 Updated copyright year to 2013 and changed method signature to return native response object instead of native request object . 
 Remove unnecessary logger statement and fix FIXME comment in HttpProtocol . java
 Added methods useCaches ( boolean )  and acceptEncoding ( String )  to HttpRequest class . 
 Added TODO comments and updated documentation for the Context class . 
 Refactored ExpressionTypingVisitorDispatcher to handle resolution scope for non - reference expressions . 
 Added  @ Ignore annotation to EJBClientClusterConfigurationTestCase due to intermittent failures . 
 Fix grammar and formatting in ApplicationUtil . java . 
 Refactored ClusterFactory interface to use non - final variables and removed unnecessary imports . 
 Addded org / gradle /  ,  javax / jms /  ,  ch / qos / logback /  ,  and org / apache / logging / log4j /  to the list of problematic classes in MethodDatabase . 
 Added metrics for vertex index store ,  edge store ,  and cache misses / hits . 
 Add nullability annotation for get ( Key < T >  )  method . 
 Refactored AddDialogFragment to use a single Show instance instead of creating a new SearchResult instance . 
 Fix issue with EJB container deletion by adding missing  ' tempFile . delete (  )  '  call . 
 Refactored IoActions . createFileWriteAction to use a more generic return type . 
 Added logging for gradient JAR file generation . 
 Update Connection . java to seek log reader to 0 after active index is updated . 
 Add method createSoundFromFileDescriptor to SoundFactory ,  allowing sounds to be loaded from file descriptors . 
 Fixes issue  # 1234 :  Add fork option to RunMojo to allow forking of the JVM . 
 Removed unused import statement for org . apache . camel . util . Main . 
 Add support for removing all cache entries with a given URL prefix . 
 Add nullability annotation to JetContentBasedFileSubstitutor . getDecompiledPsiFile method . 
 Fixed PathTypeCoercer to use try - catch block for runtime exceptions and added CoerceFailedException for better error messaging . 
 Modified the delete method in ODatabaseRecordAbstract to use the record ' s underlying cache instead of the database record ,  and added a check to ensure the record is not null before calling callback hooks . 
 Fix memory mapping error and reduce max memory to prevent buffer overflow . 
 Refactored EditorHyperlinkSupport to remove todo comments and optimize code . 
 Remove unnecessary messages and delay in keyguard drawn timeout
 Added assertion for output and input channel IDs in GateDeploymentDescriptorTest . 
 Reload event is deprecated and will be removed in Tomcat 8 . 0 . x onwards . 
 Add metric names to dimension exclusions in DataSchema . 
  Fix incorrect use of  '  =  =  '  operator in CameraManager . isAvailable (  )  method . 
 Update JavaQueryTest to use org . junit . Assert . assertTrue instead of deprecated JavaDocsGenerator and AsciidocHelper . 
 Updated EditorFactoryAdapter to include myEditor in editorReleased event
 fixed encoding exception for ServletRequest and ServletResponse
 Added check for NaN score and updated serialization for InternalSearchHit
 Fixes bug in PostprocessedBitmapMemoryCacheProducer by passing correct  ' isLast '  argument to consumer . 
 Override performCompletion (  )  and onKeyUp (  )  methods to avoid extra onItemClick being called and allow threshold of zero
 Fix bug in JGitUtils :  handle null or HEAD objectId in getBranchObject . 
 Modified OperationSetPersistentPresenceIcqImpl to use StatusResponseRetriever for user status query when status is not set  ( is  - 1 )  and user is offline . 
 Updated logging level for c library function detection from info to debug . 
 fixed bug in NamedScriptableDefinition . writeExternal method
 Set editing to false in IpnbCodePanel to prevent simultaneous editing and execution of code cells . 
 Remove deprecated method getColumnFamilyStore ( String )  and replace with getColumnFamilyStore ( int ) 
 Added Bean Validation Exception throwing capability to BeanValidator . 
 Added import statements for Executor and Executors ,  and replaced ForkJoinPool with Executors '  newCachedThreadPool . 
 Added method cancelTaskWindowTransition ( int taskId )  to IWindowManagerImpl . 
 Add R 2 . 14 compatibility for source files with no trailing newline . 
 Removed unnecessary imports in UnifiedParser . java . 
 Remove import statement for ZWaveNetworkEvent . 
 Adds a comment mentioning a future plan to provide information about whether a version was requested in the graph . 
 Fix issue  # 936 :  Remove unnecessary transformation method for search text field . 
 Fix issue with SystemServer . java by removing unnecessary code and updating pm . performBootDexOpt (  )  ; 
 Added support for serializing view state with base layer information . 
 Added methods getPsiFile (  )  ,  getEditor (  )  ,  and getDocument (  )  to interface HighlightingSession . 
 Added validation for action parameter in AccessibilityEvent . setAction (  )  method to ensure it ' s a valid action defined in AccessibilityNodeInfo . 
 Update getHelpId (  )  to return create_new_project_dialog and allow starting dumb mode inside show (  )  method . 
 Deletes blob file if it exists ,  otherwise throws an IOException . 
 Update import statement for AccountUtils to use common account library . 
 Added getUri method to EndpointValidationResult and updated CamelCatalogTest to include additional validation tests . 
 Update copyright year to 2016 and fix message pattern for unresolved files . 
 Modified ImageProcessor and TexturePacker2 constructors to accept a null rootDir parameter and removed unnecessary Settings parameter . 
 Updated SSLConfig to use TrustStoreConfig and IdentityStoreConfig to load truststore and keystore instead of hardcoded paths . 
 Fix log formatting to correctly handle special format characters . 
 Added a warning that the AutoValueExtension API is not final and will change in a future release . 
 Update ConfigTest to use getFile (  )  method for iOS info plist file . 
 Added import statement for PluginManagerCore and updated assert statement in getPluginVersion method to include a list of available plugins . 
 Fix BeaconTransmitter to handle null BeaconParser instance and correctly set serviceUuid . 
 Clear usage list and reset tree model to show only the first usage . 
 Allocator :  Changed block store location view from int to String
 Added CDDL header and updated copyright notice for TroffAnalyzerTest . java . 
 Started DNS service and registered ParallelResolver and DnssecResolver . 
 Register Desktop as the means by which Neo4j was launched . 
 Fix initialization of boxes to properly handle infinite bounds . 
 Added  ' setContentIntent '  to UserSwitcherController for removing guest user and logging out
 Added check for fullscreen mode in sanitizeOverrideConfiguration method to avoid overriding global config with empty override config . 
 Added writeable callback to AsyncSSLSocketWrapper to enable buffering of data written during handshake . 
 Added import statement for ApplicationInfo and modified code to use PackageManager . GET_UNINSTALLED_PACKAGES flag to get resources for uninstalled packages . 
 fixed null pointer exception in getImage method by adding null check before creating ImageIcon
 Remove unused constants TITAN_HADOOP_GRAPH_OUTPUT_TITAN and TITAN_HADOOP_GRAPH_OUTPUT_TITAN_INFER_SCHEMA . 
 Renamed Method . total to Method . min and fixed a bug in JSONModel . load where it was using the wrong primary key for the query . 
Remove unused imports in AbstractBindingWriteHandler and BindingMetricHandlers . 
 Added import statements for ErrorProneScanner . EnabledPredicate and default checks ,  and conditionally set the Scanner . class context to ErrorProneScanner with default checks . 
 Fix compression test to correctly check for stream compression usage . 
 Fixed PostgreSQL array handling in DefaultBindContext by changing the method signature to accept an Object [  ]  instead of an Object
 Added a check to ensure that the EntryPoint extension point is properly registered or unregistered during test setup and tear down . 
 Remove unnecessary  ' accept '  method in GrVariableImpl
 Update CreateSwitchIntention to extend BaseElementAtCaretIntentionAction and import the necessary class . 
 Updated kill command formats to use  % s instead of  % d for session and statement killing . 
 Increased MAX_DURATION from 10 seconds to 60 seconds . 
 Add import statement for Arrays class and modify copyOf method to use Arrays . copyOf . 
 Fixed trimming of cell values to remove leading and trailing whitespace . 
 Fixes issue with image resource not updating when resource configuration changes . 
 Added method getServiceConfig ( String )  to return ServiceConfig for a given service name . 
 Remove unnecessary code for suspending incoming traffic
 Added missing  ' } '  to test case for try - finally block with break statement . 
 Fix issue with cycles in UniversalEnglishGrammaticalStructure . 
 Modified BluetoothManagerService to handle error message for dump command . 
 Added nickname chat room menu item . 
 Fixes Javadoc and adds rational for UpperEllCheck . 
 Fixed VcsLogDateFilter to match commits by commit time instead of author time
 Fix IllegalArgumentException message to include image dimensions . 
 Remove entry from session cache and mark dirty if it existed . 
 Updated console output format for  ' orient db console '  command . 
 Fixes SPR - 13139 :  Improved wildcard handling in AntPathMatcher . 
 Update HeliosSoloIT to use  ' apk add  -  - update '  instead of  ' apk - install '  for installing bind - tools . 
 Added import statements for Matchers ,  WrapUtil ,  and Hamcrest ,  and removed unnecessary assertions . 
 Fix issue with unnecessary XposedBridge log statement . 
 Removed unused variable  ' dragIcon '  and updated the transfer actions supported by the source . 
 Fix bug in HttpMessageDecoder :  handle non - empty header lines correctly . 
 Added import statements for javax . persistence . Temporal and javax . persistence . TemporalType ,  and added the  @ Temporal annotation to the getTimestamp (  )  method . 
 Fix JavadocStyleCheck to allow  @ inheritDoc with throws clause . 
 Initialize new component and add route to routing table . 
 Fix for issue  # 12345 :  Carrier name was not being properly set in DataUsageInfo
 Replace method reference with context - dependent cached value provider to prevent recursion . 
 Add error handling for unexpected errors during index updating . 
 Add ignore annotation to JMXMonitorTypeLongCounterTest due to missing monitorNotificationLong . xml file . 
 Added support for annotating plain text files . 
 Fixes  # 23456 :  Adds ability to specify libraries to unpack from fat jars at runtime . 
 Forbid shared user change during package installation . 
Added stopAllSounds method to AudioManager and called it in World ' s close method to stop all audio instances when the world is closed . 
 Added method getArg ( int index )  to FeaturesTest class . 
 Seal method updated to close inverted index creators and added a check to only iterate over columns with inverted index creation enabled . 
 Fixes issue with negative count in index hits from querying removed nodes . 
 Refactor PropertyEventBus . java to remove unnecessary code and improve efficiency . 
 Remove unnecessary import statement for CheckstyleException . 
 Fix CoyoteAdapter to properly handle async requests and remove unused code . 
 Update device properties in AbstractInstrumentationMojo and Test classes to reflect Android Maven Plugin instead of Maven Android Plugin
 Fix assertions in GitQuickListContentProvider . 
 Fixes issue  # 3003 :  Implemented manual call to endTextEditHistory (  )  before starting a new CompoundEdit to avoid messing up auto - format undo . 
 Update Groovy compiler customization builder documentation link from codehaus . org to docs . groovy - lang . org . 
 Added a check for default constructor availability in JsonParserUtil . 
 Refactor TachyonException to check and downcast derived exceptions before throwing them . 
 Added transitory tags to empty task data
 Added a scrollValuesObsolete flag to LightGrid to indicate when scroll values are outdated . 
 Add option to disable DOT file generation in RunCamelMojo . 
 Modified EventBusBasicTest to use MyEvent class and updated assertions . 
 Added test for context . getRealPath (  )  with platform - specific file path
 Updated LanguageAction to include Unix in the confirmation message
 Update ManagementCenterService to use new management center URL . 
 Removed unused method getKeepAliveLeft (  )  and updated related methods to use the corrected field name  ' keepAliveLeft '  . 
 Added check for channel registration before setting executor in DefaultChannelHandlerContext . 
 Fixes issue with expire store size not being set properly in CacheOffHeap . 
 Reverse logic in VanillaInventoryCodeHooks to return true if tile entity has no item handler capability . 
 Adds a reference to the OnMethodEnter # skipOn (  )  method for advising on method enter . 
 Changed log level from debug to error for better error handling in CommandContext . 
 Remove unused enum constant NonDefault and simplify DisableCheckSpecialChar . 
 Add MathJaxLoader . java ,  a file containing the copyright and license information for the MathJax library used in RStudio . 
 Fix warning message for Audio clip uses an external  . resS file that doesn ' t exist . 
 Update SplitFrame to use Key < Frame >  instead of Key [  ]  for destKeys
 Update MagicConstantInspection to extend BaseJavaLocalInspectionTool instead of LocalInspectionTool . 
 Update WXSwitch to use ToggleButton instead of Switch and import TypedValue for text size modification . 
 Fix invoke method to return value instead of invoking directly . 
 Adds a new method to FingerprintManager that throws IllegalArgumentException if the crypto operation is not supported or is not backed by Android Keystore facility ,  and IllegalStateException if the crypto primitive is not initialized . 
 Fixed bug in Intent . java by adding a check for mAction  !  =  null before exposing file URIs . 
 Added garbage collection to free up file locks and allow Windows to manipulate library files . 
 Refactor JBTable to ignore non - typed events and undefined key characters in autoStartsEdit mode . 
 Remove hack for class file v53 support in JavaPluginConvention . 
 Increased throttling delay from 200ms to 750ms in ThrottlerTest . 
 Added WebSettings . LayoutAlgorithm import and setLayoutAlgorithm method to TestWebSettings class . 
 Add method toString (  )  to class Crouton . 
 Committed changes to the contact list service ,  including a new error code for when a contact already exists in the list . 
 Fixes issue with formatting of decimal values in ItemUIRegistryImpl . 
 Remove unused  ' matchesAnyOf (  )  '  method and add nullability annotations to  ' apply (  )  '  method in CharMatcher and Function classes . 
 Update ResourceModel to use void setCopyResourceId ( Integer )  instead of ResourceModel setCopyResourceId ( Integer )  and return this ; 
 Added support for checking unused imports in enum and annotation definitions . 
 Add ensureYellow (  )  call to TransportSQLActionTest . testSelectFrom_Doc (  )  to ensure yellow shard is created . 
 Added import statement for IndexMetaData and updated createIndex method to use prepareCreate with settings . 
 Remove unused import and code related to UsageStatsManager
 Added tests for escaping and quoting inside String literals in EL expressions . 
 Fix parsing of sequence flows with non - unique IDs . 
 Update BloomFilter to use ColorRGBA . BlackNoAlpha for better performance . 
 Added import statement for `com . tinkerpop . gremlin . process . T` and updated `tx . addVertex` method to use `T . label` instead of hardcoded string event for label name . 
 Added logging for unknown drawer states . 
 Modified DnsEventListenerService to log only as many events as actually occurred . 
 Update BaseIndex . java to change log level from info to debug for insertion
 Fix bug in Param . java by correcting the bitwise OR operator in the for loop . 
 Add reinit method to reset presence panel values . 
 Add custom LoadMoreView and swipe custom LoadMoreView methods to UltimateViewAdapter . 
 Added method to dump alignment for block . 
 Added test method testIt (  )  to class TestJsonPointer . 
 Refactor :  Move maxHeight calculation to a separate method and use it in both places . 
 Updated LoginManager to check presence status before activating account and registering provider . 
 Fixed a bug in the database cache by using a different ID for the file history . 
 Fix bug in XmlPluginDescriptorReader :  changed return value to use  ' pluginXml '  instead of  ' pluginDescriptor . createRelative (  .  )  '  to ensure correct plugin directory is used . 
 Fixes CAMEL - 1234 :  Improved error handling in FromFileDoNotDeleteFileIfProcessFailsTest to correctly handle retries and minimum message count . 
 fixed DeferredIconImpl to use ScalableIcon ' s scale method when scaling icons . 
 Add method gc (  )  to class System ,  which calls Runtime . getRuntime (  )  . gc (  )  to perform garbage collection . 
 Add ability to create and manage run configurations for a project . 
 Remove constructor and set numRecordsPerBlock to a constant value
 Added fallback logic to create a new block store if loading from file fails . 
 Added CountDownLatches for tracking container allocation and application completion
 Added check for FoldRegion validity before setting expanded state . 
 Fix XML serialization by changing id to dc : identifier in book element and added error handling for SAXException . 
 Updated SettingsMenu to use CTRL_ENTER_COMMAND instead of ENTER_COMMAND for sending messages . 
 Fix for IDEA - 227428 :  Handle disposed modules in MavenProjectImporter . 
 Refactor LayoutManager to handle section layout with variable height . 
 Remove unnecessary import statement and add missing parameter to adapter constructor . 
 Added handling of ClientAbortException during response flushing in StandardHostValve
 Added a release call for the script source to prevent memory leaks . 
 Update WebSocketServerFactory to use ServletWebSocketRequest and ServletWebSocketResponse instead of deprecated ServletUpgradeRequest and ServletUpgradeResponse . 
 Updated ClassLoaderLogManager to use its own class name in DEBUG_PROPERTY constant
 Modified CassandraUnitUtils to use a specific port  ( 30000 )  for the embedded Cassandra server . 
 Add missing unique constraint and index changes to the diff result . 
 Added variable  ' distance '  to SavingTrackHelper and set it to 0 ;  no other changes . 
 Remove unnecessary dependency management info for version mediation of transitive dependencies
 Added check for empty active notifications map before showing individual notification . 
 Refactor CodeStyleSettingsCodeFragmentFilter to use Collection < String >  myAllFields and add method stop (  )  for CompositeSequentialTask . 
 Removed unnecessary import statement for  ' List '  in DnsQueryContext . java . 
 Fixed ClientEncodingFilter to correctly handle encoding by returning false instead of true
 Added method to create a new task that fails if the original task is not completed within a given timeout
 Remove unused import statement for FileTrackingEvent
 Fix issue with upload client exception handling and enforce creation of a new client object for next uploads . 
 Added RIGHT_MARGIN option to Groovy Language Code Style Settings Provider and changed order of options in Python Language Code Style Settings Provider . 
 Fixes duplicated entries for work - only apps on multi - user devices by removing uninstalled entries for the owner user . 
 Modified StripCode . java to always strip types that defined on a type that is being stripped and added a TODO for further simplification . 
 Added logging for received raw messages from CUL . 
 Remove unnecessary override of MethodInterceptorStub . intercept (  )  method . 
 Fixes SERVLET_SPEC - 44 :  Changes the logging message when a connection is terminated due to a failure to send a message to the client ,  to include a reference to the relevant bug ticket . 
 Add import statement for EventRequest and modify unsetPausedIfNeeded method to check for EventRequest . SUSPEND_EVENT_THREAD policy . 
 Changed component model plugin descriptor to use simple name instead of fully qualified name . 
 Added getClass (  )  . getSimpleName (  )   +  traversal detail to FilteredTraverserBase . toString (  )  method . 
 Optimize HpackEncoder by using a custom data structure for dynamic table to reduce allocations . 
 Fix grammar :  change behavior to behaviour in HTTP client definition . 
 Fix issue with line width calculation in BitmapFont . java by removing unnecessary code and simplifying the calculation . 
 Fix EnterAction . java :  Remove unnecessary  ' else '  branch and simplify code . 
 Added DOWNLOADED_START case to BroadCastManager to handle download start events . 
 fixed bug in ResolverActivity where it was killing itself due to strict mode violation . 
 Adds a new method collapse (  )  to collapse all expanded items and a new method collapse ( int position )  to collapse a specific item . 
 Updated InjectableObjectFactory to use LinkedBlockingDeque for pushBackInjection instead of LinkedList . 
Remove unused import statement for SQLiteStatement . 
 ImapConnection . java :  Add sensitive logging for IMAP commands . 
 Fixed encoding in NagiosEncryptionMethod by replacing  ' TripeDes '  with  ' TripleDes '  . 
 Reduced timeout for ApplicationImpl ' s thread pool from 30 minutes to 5 minutes
 Renamed SerializedConverterTest . java to QueueFileLoadingTest . java and migrated tests to use QueueFile directly . 
 Fix error handling for asset upload errors ,  including dismissing the error description dialog and adding tracking events for too large files and recording failures . 
Added method init ( Context context )  to initialize DBFlow and retrieve Database holder via reflection ,  and added a call to getDatabaseHolder (  )  in the constructor . 
 Add PKCS  # 12 key archive media type . 
 Added setting listener to set enum class for preference impl . 
 Refactored initCalendarAndGroups to use AsyncTask for background processing and added initialization of ListPreference for calendar and group syncing . 
 Refactored handling of share permissions updates to simplify code and improve error handling . 
 Added  ' isRejoining '  method to CollabEditStartParams class to indicate whether collaboration session is rejoining an existing session . 
 Refactor FilterStreamInput to be abstract and add a private final delegate field . 
 Fix typo in error message . 
 Fixed issue with cache management in DrawTask by correctly setting repositioned variable . 
 Added support for forcing HTTP requests to be intercepted by ZAP . 
 Added test cases for stubbing with throwables using  ' fail '  method . 
 Add handling for indexShardRoutingTable being null in PlainOperationRouting . 
 Fixes  # 1234 :  Ensure that super classes are properly resolved when they are not in the scope of the containing file . 
 Changed intent action from ACTION_PICK to ACTION_GET_CONTENT for video library selection . 
 Add Linkify . WEB_URLS and Linkify . EMAIL_ADDRESSES to NavUtil . java . 
 Added a `closed` field to the SyncingFileSystem class to track whether the output stream has been closed ,  and added a check for this field in the close (  )  method to avoid syncing the file twice . 
 Add import statement for Providers class and bind DynamicSettings to null provider to prevent Guice errors . 
 Update message collector to use plain text instead of System . err
 Add CSVResultsExtractor parameter includeId and update CSVResult constructor accordingly . 
 Fix FileSystemMasterIntegrationTest to use mLocalTachyonClusterResource . get (  )  . getMaster (  )  . getInternalMaster (  )  . getFileSystemMaster (  )  instead of mExecutorService . 
 Fixed issue with user permissions not being updated correctly by using the user ' s permission set instead of the form data ' s permission set . 
 Fix issue with PullToRefreshBase class where width and weight were incorrectly used . 
 Fix animation end event handling to prevent duplicate calls to onAnimationEnd listener . 
 Modified SingleRootFileViewProvider to use myDocument instead of getDocument (  )  for getting modification stamp . 
 Refactor BaseTopActivity to use abstract method for firing tracker events . 
 Adds a check for multiple contention actions for a given lock ,  throwing an UnsupportedOperationException if found . 
 Added test for integration with spring - integration - dsl - groovy - core 1 . 0 . 0 . M1 . 
 Fixes issue with PanelView ' s touch slop exceeded flag not being properly set . 
 Removed unnecessary code and optimized bubble removal logic . 
 Fix issue with null module reference in PluginRunConfiguration . 
 Added methods onRemoveBlockByClient and onRemoveBlockByWorker to LRUEvictor to implement eviction logic . 
 Added try - catch block for FileNotFoundException to handle missing distance model file . 
 Added deleteOnExit method to File class ,  including synchronized deleteOnExitList and registerDeleteAction . 
 Replace polygon triangulation algorithm with a better one using glu tesellator code from PGraphicsOpenGL . 
 Remove getLogger (  )  method and use a private constructor to enforce singletons . 
 Fixed bug in BarController . java by changing View . SYSTEM_UI_FLAG_FULLSCREEN to View . SYSTEM_UI_TRANSPARENT in the if statement . 
 Added import statement for com . google . common . base . Throwables and updated recoveryFailed method to use Throwables . getRootCause . 
 Update ContributionManagerDialog to show a progress message while downloading the contribution listing . 
 Fixed typo in test method name from  ' equans '  to  ' equals '  in HttpHeadersTest . java
 Extracted MemberType class from ResolutionResultsCache and added constructors ,  debugName and toString methods . 
 Added OsmandExtraSettings and OsmEditingPlugin ,  removed OsmandRasterMapsPlugin . 
 Added import statement for ActionMethodParamNameResolver and registered it as a component in WebApplication . 
 Clear view cache by adding a clearCache method . 
 Added integration branch substitution to ticket page . 
 Removed unused imports GWT ,  ClientBundle ,  and ImageResource in GlobalToolbar . java . 
 Add WebListener annotation to ModelManagerListener and remove unnecessary application listener registration . 
 Fix bug in ReadOnlyStoreManagementServlet by removing unused variable and method call . 
 Removed unused method  ' itemsFormat ( JsonFormatVisitorAware toVisit )  '  from ArraySchemaFactory and JsonArrayFormatVisitor interfaces . 
 Refactor InternalAttribute . equals method to use InternalAttribute instead of Attribute
 Fix user permissions configuration to handle null or empty input . 
 Replace hardcoded API keys with placeholders for user input . 
 Update WPWebChromeClient to check for empty title before updating activity title ,  and also check for null activity . 
 Add test for outer join with common expression . 
 Modified ActivityManager . java to add flag MOVE_TASK_NO_USER_ACTION to moveTaskToFront method . 
 Added test method `testFormat` and renamed `format` to `testFormat` ,  `parse` to `testParse` ,  and `cloneObject` to `testCloneObject` to conform to Java naming conventions . 
 Added Apache License header to CommandUtils . java . 
 Added VisibleForTesting annotation to clearLoadedModules and getLoadersMap methods and updated the signature of getClassLoaderForExtension method to use the updated map . 
 Fixed bad URL in HttpClientTimeoutTest to use  ' badport '  instead of hardcoded port number . 
 Adds a new abstract method featureValueOf ( T actual )  to the FeatureMatcher class ,  allowing subclasses to extract the interesting feature from the target object . 
 Refactored method signature for DefaultStructBindingsStore . collectManagedProperties to remove unnecessary type parameter D . 
 Refresh files after rebase . 
 Modified RoutePreferencesMenu to use routingHelper . getAppMode (  )  instead of settings . APPLICATION_MODE . get (  )  for getting application mode . 
 Refactored code to use character classes instead of hardcoded whitespace characters . 
 Fix AppBarLayout . java to correctly calculate mDownPreScrollRange . 
 Modified StreamBody . java to pass stream length to Util . pump (  )  for better buffering . 
 Hide filter radio group and show sign - in button in empty view
 Added check for disposal before running task in AbstractTreeUi
 Updated BundleReceipe to reload dashboard and update widget positions . 
 Added import statements for SeriesGuidePreferences and R ,  and updated the ListWidgetConfigure class to use the imported classes and notify the app widget view data changed . 
 Removed unnecessary  @ GuardedBy annotation from SubscriberRegistry . 
 Fix bug in TcpFailureDetector :  change `removeSuspects . get ( m ) ` to `removeSuspects . get ( m )  . longValue (  ) ` to avoid potential ClassCastException . 
Implemented confinement to stages in ManagementGraphIterator . 
 Fixes  # 42 :  Remove unnecessary ExecutorService finalizer . 
 fixed GPXLayer . java to not display points for empty routes
 Added method getPreferredSize (  )  to BranchesPanel to handle visibility and size . 
 Updated SocketConnection to include additional debug information in log messages . 
 Added import statement for Logger and added logging statements for debugging purposes . 
 Refactor AbstractAnnotationProcessor to exclude JDK classes from superclass checks . 
 Allow null path info in HTTP requests . 
 Increased ID authority wait time from 10ms to 100ms . 
 Added method getRootFile ( String pathname )  to retrieve the canonical file for a given pathname ,  and updated the storeDir and migrationDir variables to use this method . 
 Removed IterationListener interface implementation from OutputLayer
 Added license header and updated copyright notice in IndexSynchronizer . java . 
 Fixes bug where stack was being moved even though it was not added ;  adds log message for stack move failures . 
 Fix bug in FileHandler where writer . flush (  )  was called unnecessarily . 
 Fix pool hard cap violation exception message to use a question mark and add a space after  Hard cap  =   . 
 Fixed infinite loop in NioEndpoint by replacing while  ( running )  with while  ( true )  and adding a check for destroy (  )  called . 
 Fix issue with drop recipient ANRed and improve handling of invalid drop - result claims . 
 Added timezone information to SystemResource and ServerStatus . 
 Added immutable map for services in FileSystemWorker
 Fix circle drawing method to use drawCircle instead of drawPath for outer circle . 
 Added test case for verifying no interactions wanted in order . 
 Rename  ' setType '  to  ' setDataType '  in OracleTableColumn class and update the signature to reflect the change . 
 Added closing of fsDataStream and fs to ensure proper resource handling in BSONSplitter . 
 Fixed issue with content view pivot not being set correctly when selecting a bubble . 
 Replace throw new RuntimeException with throw new IllegalStateException in ByteArrayClassLoader . 
 Modified MethodLibrary to use ArrayList instead of List and added missing  ' ensureCapacity '  calls to avoid unnecessary array resizes . 
 Added method getWorld (  )  to interface Chunk to retrieve the parent World . 
 Fixes issue with hidden providers not being accounted for in GlobalStatusSelectorBox creation . 
 Fixed issue with removeChangeListener method by properly checking for null before attempting to remove listener . 
 Added test cases for BitmapFactory . decodeResource (  )  and decodeFile (  )  methods . 
 Replace ISODateTimeFormat . getInstance (  )  with ISODateTimeFormat . getInstance ( chrono )  to use the specified chronology . 
 Add logging statement in Webapp . java to warn about development mode being enabled . 
 Adds method to RequestContext to record filter execution summaries ,  including name ,  status ,  and execution time . 
 Add exception message with POJO name for better error handling . 
 Fixes issue with finalizer being called on uninitialized mNativeCanvas . 
 Updated ViewPagerActions to change the description of moveOnePageToTheRight to move to a specific page . 
 Added test case for sorting list of strings with custom comparator . 
 Add editRmdFormatOptions command to TextFileType and make it visible and enabled when RMarkdown package is available . 
 Add test case for JSONPObject :  test_jsonp_1 . 
 Update ExopackageDexLoader to use stale instead of state in logging message for deleted ODEX files . 
 Adds a custom dispatchSetPressed method to prevent ViewGroup from dispatching pressed state to its children . 
 fixed issue with artifactId formatting for SignavioFileSystemConnector . 
Fix crash and backup issues for Spanish ,  Polish languages . 
 Renamed constructor parameter from verticalDivider to divider to support horizontal dividers . 
 Remove unnecessary imports  ( JCheckBox ,  JComponent )  and updated import statements . 
 Update ReactTextShadowNode to use ceil instead of rounding for calculating image size
 Modified ChannelService to use target . addService instead of AsynchronousService . addService . 
 Bump libgdx version from 0 . 9 . 9 to 1 . 0 . 0 . 
 Added check for root id existence before calling isRootId (  ) 
 Fix issue with BitmapFactory . decodeStream returning bitmaps with null configs by adding check and handling for null configs in LruBitmapPool . 
 Add FML import statements and annotate BlockHighlightHandler methods with  @ SideOnly and  @ SubscribeEvent for Forge mod compatibility . 
 Refactor code to remove unnecessary SDK version checks and use registered lifecycle callbacks for push notifications backend monitoring . 
 Added check for redeclaration diagnostics and marked them with an error annotation . 
 Added permission check for scheduling jobs with FLAG_WILL_BE_FOREGROUND
 Fix Peer . java to use `addr . toString (  ) ` instead of `address . toString (  ) ` for consistency . 
 Clear separators position consumer and reset markup . 
 Fixed issue with incorrect display of numeric values in ResultSetViewer by using the correct data type and handling null values . 
 Update CountDownTimer to use mCountDownFormatter when displaying undo . 
 Added import statements for CleanupHelper and modified CliTest to extend CleanupHelper
 Added check to skip firing modules added event when no modules are added
 Fixes issue with null pointer exception in checkIfCameraClosedOrInError method . 
 Fixes issue with ViewGroupOverlayUtils ,  now properly handles screenX and screenY coordinates . 
 Remove unused import statement for  ' containsLine '  in BuildScriptErrorIntegrationTest . 
 Added tests for bug 54602 :  TestCoyoteAdapter . testBug54602d and TestCoyoteAdapter . testBug54602e . 
 Adds a new method `isES (  ) ` to check if the current OpenGL version is an OpenGL ES version . 
 Enable ShowUsagesAction only when there are usages to be shown . 
 Added logging for collapsed panel in keyguard state . 
 Refactored IgnitedAsyncTask to use single argument execute method and removed unnecessary execute (  )  method . 
 Increase heartbeat frequency from 5 seconds to 30 seconds . 
 Deprecated ConfigAttributeEditor class to help detect hard - to - spot configuration issues for end users . 
 Remove unnecessary intent . 
 Refactored VcsLogTabsWatcher to use ToolWindow instead of ToolWindowImpl . 
 Add import statement for JavaResolveUtil and modify FunctionalInterfaceSuggester to use it . 
 Fix error message to correctly reference NIO constraint level . 
 Modified AnalysisScope to check for test source content before processing files . 
 Modified PresenceFilter to always show chat rooms and recent messages . 
 Updated AxisReportIncidentServiceTest to use  ' localhost '  instead of  ' 127 . 0 . 0 . 1 '  for the connector host . 
 Fixes issue  # 1234 :  Changed await to assertOpenEventually in BasicMapTest to use a more appropriate method for testing latches . 
 Update FileEditorManagerImpl to use UIUtil . getLabelForeground (  )  instead of Color . BLACK for unknown file status colors . 
 Adds the  @ since 2 . 5 annotation to the SentenceWhitespaceRule class . 
 Update CommentDetailFragment to show progress bar while submitting comment . 
 Update log message for creating database to using database
 Fix quit method to use message instead of callbacks . 
 Fixes  # 12345 :  Add test for mapReduce output collection and fix getOutputCollection (  )  method to return the correct collection . 
 Remove unnecessary fragment instance creation
 Set writable flag for virtual file based on delegate ' s writability
 Refactoring :  Changed visibility of introduce constant popup to false . 
 Added method getPreferredSize (  )  to JBLoadingPanel ,  returning the preferred size of the content panel . 
 fixed error message in SQLCreateIndexTest . java
 Refactored ProjectRegistry to use a while loop to iterate over relative path segments and find a matching ResourceHandlerDescriptor . 
 Fix PythonTask . java to correctly set up bash environment on non - Windows platforms . 
 Removed unnecessary field and methods ,  moved getExpectedResolveData (  )  to doTest (  )  method . 
 Added constructor for Time class to allow for Jid parameter and updated EntityTimeManager to use it . 
 Fix bug in ContactInfoGenerator :  handle empty URLs . 
 Fix SipPhone equals method . 
 Initialize number of allowed radio channels when Wi - Fi is turned on . 
 Make static serializer instances final
 Set MAXFILESIZE to 64000 as a final constant . 
 Replace termCapacity with termBufferLength in publicationTermWindowLength and ipcPublicationTermWindowLength methods . 
 Add test for large requests ,  including printing total memory usage and adding a ignore flag to prevent test failure . 
 Added import statement for org . junit . Assert and added method getOnlyConstructor to TestClass . 
 Fix crashing issue with ReactTextView by setting non - null LayoutParams . 
 Added import statement for com . intellij . lang . ASTNode . 
 Modified OServerCommandGetStaticContent to return true instead of calling super . beforeExecute . 
 Fixed issue with empty stop array in PlanGenerator . 
 Added import statement for com . intellij . ui . treeStructure . Tree . 
 Adds IsoChroneSPTRendererAccSampling and IsoChroneSPTRendererRecursiveGrid to the list of bound classes in OTPConfigurator . 
 Update GLRenderer to set viewport to reflect framebuffer ' s resolution before binding framebuffer . 
 Fixes  # 12345 :  Add test for double - decoding of XML entities . 
 Added method getConfusionMatrix (  )  to Evaluation class to return the confusion matrix variable for this evaluation . 
 Fix formatting issue in FormatterUtil . java by reversing the logic of the containsWhitespacesOnly method . 
 Fixed issue with non - iterable types being passed to QueryImpl
 Update VolumeUI to use PendingIntent . FLAG_UPDATE_CURRENT when creating notifications . 
 Update Statistics . java to increment numObjects by 1 instead of count
 Fix BukkitUtil toLocation method to use Location instead of TeleportLocation . 
 Refactored HystrixTimer . reset (  )  to use null - safe getAndSet (  )  and added idempotence for shutdownNow (  )  to prevent NPEs . 
 Modified PrintActivity . java to use Extra_Print_Document_Info instead of Extra_Print_Job_Info for passing print job information to the print service . 
 Remove unnecessary `db . beginTransaction (  ) ` and `db . endTransaction (  ) ` calls in ReaderAttachmentTable . 
 Added tearDown method to remove persistent interval data after testing . 
 Refactored import statements and removed unused imports . 
 Disable content loading for unknown file types
 Added support for retrieving annotation attributes from meta - annotations . 
 Added NullOutputStream . java ,  a class that throws away all output instead of outputting anything ,  based on code from Stack Overflow . 
 Fix Sonar server rule index to properly handle empty query text . 
 Remove unused import statement for tachyon . master . IndexedSet . 
 Modified ZWaveMultiLevelSwitchConverter to correctly handle 99 %  value as 100 %  in OH . 
 Fixes HIVE - 22276 :  Add null safety check for join keys to prevent generation of unnecessary IS NOT NULL predicates . 
 Modified the BaseCppTest class to use clang +  +  instead of the default C +  +  compiler and added a try - catch block to handle exceptions . 
 Fix bug in Patterns . PHONE regular expression to allow for international phone numbers . 
 Added missing closing parenthesis in SelectQueryRunnerTest . 
 Added attribute service to set initial attributes for the root directory . 
 Added a check for a valid ID in the ScriptIntrinsic constructor to prevent loading of invalid scripts . 
 Added method toString (  )  to FirebaseMessage class . 
 Fix FacebookSdk to handle app Ids correctly
 RoutingHelper :  Ignore route recalculation if parameters haven ' t changed . 
 Adds a clearActor method that removes the current actor for the cell and returns the cell . 
 Added return statement at the end of the doSurround method to ensure proper function return . 
 Added SecurityCheck annotation to UserREST . register method to enforce role - based access control . 
 Fix issue with concurrent unsubscription handling in OperatorPublish . 
 Added method getAll (  )  to GHDirectory to return all created directories . 
 Added ISODateTimeFormat import and updated timestamp parsing to use Joda - Time ' s parseDateTime method . 
 Update TestNGUtil to use relative path for ice_root . 
 Fix dock state extra field in Intent . java . 
 Fix for security vulnerability :  Check validity of user before clearing calling identity . 
 Fixes issue  # 1234 :  Add  ' final '  keyword to variable  ' instance '  in method  ' validateCheckSection '  to prevent accidental changes to the variable . 
 Fix WebView scrolling by considering visible title height . 
 Added SpringCamelContext integration and PropertiesParser bean . 
 Fixes  # 1234 :  Added trim (  )  function to remove unnecessary semicolons in the SQL files . 
 Added static import for REPLACE_EXISTING and updated Files . copy method to use it . 
 Commit message :  Fix issue with missing deserializer for property values . 
 Remove  @ GwtCompatible ( serializable  =  true )  annotation . 
 Add settings for performance logging and legal terms of service URL . 
 Fixed typo in class name from  ' Characoters '  to  ' Characters '  and updated references in  ' ClassicStyleFragment '  . 
 Fix bug in NativeMetadata where empty columns were not properly handled . 
 Fix issue with unchecked warning in StoreDefinitionsMapper . 
 Added import statement for reactor . Publishers and replaced return type of handleResult method from Publisher < Void >  to Publishers . completable . 
 Remove unused JdbcSqlStat entries and add a check to reset the stat if execute count is 0 and running count is 0 . 
 Increased shutdown timeout to 30 seconds to avoid false positives in BlockingHttpEndpointTest
 Renamed AccountHelper . java from util package to models package and updated import statement and method names to reflect the new package name . 
 Modified CodegenTestsOnAndroidGenerator to skip non - kotlin files . 
 Modified NameAnalyzer to not crash when encountering GETELEMs with a global root node . 
 Added support for formatting Javadoc comments with arguments in JavaWriter ' s emitEndOfLineComment method . 
 Refactor SoftWrapDocumentChangeManager to only advance non - affected soft wraps . 
 Merge method now uses varargs to accept an array of FileBackedCache objects ,  allowing for more flexible calling . 
 Fixes issue  # 1234 :  Added support for if - then - else without curlies . 
 Added getHttpContext (  )  and setCookieStore (  )  methods to AsyncHttpClient to access and set fine - grained settings for requests . 
 Fix console scroll pos jump on first typing by adding editor . onCursorChange (  )  ; 
 Remove  @ Test annotation from testAutoCommit method . 
 Adds a new method standaloneSetup ( DefaultMockMvcBuilder builder )  to the RestAssuredMockMvc class ,  allowing for the creation of a MockMvc instance using a provided builder ,  and sets up the REST Assured mock MVC using the given WebApplicationContext . 
 Update copyright year in EnvironmentObjects . java to 2012
 Add import statement for  ' Model '  and extend  ' BuildInvocations '  and  ' ProjectPublications '  interfaces with  ' Model '  interface . 
 Added a suppression for a warning about unchecked casting in the DefaultAccessTokenConverter class . 
 Fix synchronization issue in ZWaveController . java by removing unnecessary synchronized blocks . 
 Modified UnicastSenderTest to add a comment and increase advance time to allow for 0 length data packet to be sent . 
 Changed scrollbar style from SCROLLBARS_OUTSIDE_INSET to SCROLLBARS_INSIDE_OVERLAY for the ListView in FolderList . java . 
 Add introduceConstFix for on - the - fly inspection . 
 Add integration test property to differentiate regular tests and  @ IntegrationTest tests . 
 Fixes issue with infinite loop in Roster . waitUntilLoaded (  )  method by adding a check for packet reply timeout . 
 Modified MongoCollection and MongoDatabase interfaces to add a note about binary compatibility . 
 Added logging and visibility to Console in jOOQAbstractTest
 Added public accessibility for Upsource to InspectionEngine . inspectEx method . 
 Fixed bug in NettyAsyncHttpProvider :  updated dynamic buffer size calculation to use `lenght` variable instead of hardcoded value . 
 Fixed issue with drawing images at incorrect size when using PGraphicsPDF . 
 Fix reference leak in ReferenceQueue by introducing a sentinel to avoid reference . queueNext  =  reference
 RestoreOperation :  Implemented file history and versioning for restore operation . 
 Refactored code to handle anyFunctionAggregates in OCommandExecutorSQLSelect . 
 Add a suppress warning for  @ SuppressUnderAndroid in MacHashFunctionTest . 
 Touch config to avoid lock operation in JDK 6 . 
 Modified ManualRangeMarker to return boolean value for isValid (  )  method and updated applyEvents (  )  method to void return type . 
 Removed unnecessary method createToolTip (  )  and updated getToolTipText (  )  to return null instead of an empty string . 
 Fix Windows process management to correctly handle disabled process launching . 
 Removed unnecessary import statements and removed code for requesting window features and setting flags for full screen . 
 Modified WindowState class to include new fitToDisplay logic based on task and layoutInParentFrame . 
 Added fields for customizing GWT application configuration :  width ,  height ,  stencil ,  antialiasing ,  fps ,  rootPanel ,  canvasId ,  log ,  useDebugGL . 
 Remove unnecessary import statement for org . joda . time . DateTime . 
 Fix bug in SelectStatement . java :  Cannot use IN operator on column not part of the partition key . 
 Remove enabledByDefault method and change Visitor constructor to take ProblemsHolder as parameter . 
 Enable synthetic links for WebSettings . 
 Renamed constant TABLE_MODEL_NAME to TABLE_MODEL_NAME_ in TableModelFileWriter . java . 
 Removed debug logging and simplified HandlerCaller creation . 
 Updated App URI to use underscores instead of hyphens in package name . 
 Removed unnecessary import statement for  ' Collections '  in EipAnnotationProcessor . java . 
 Fix corrupted class file handling in ClsFileImpl . 
 Added support for creating classes with a super class and updated the output message to include the class ID . 
 Remove unnecessary  ' equals '  and  ' compare '  methods from CommitLogSegment class . 
 Removed unnecessary onDestroy (  )  method and updated onResume (  )  method . 
 Set XmppConnection . DEBUG_ENABLED to false . 
 Fix issue with handling of drawException . 
 Fix highlighting data regular expressions to support escaped quotes and additional attributes . 
 Enable default off rule for rule ID  [ ruleId ]  in LanguageTool . 
 Fix bug in DataInputStream . java :  handle 0xb0 and 0xc0 correctly . 
 Add test ignore flag for test that takes a long time to run . 
 Dispose of the old root model and replace it with a new one . 
 Adds a suppress warning annotation to the makeAnnotation method to avoid unchecked warning . 
 Added display of currently active wake - up interval in the dialog asking user to keep or stop navigation service . 
 CommitAllDocuments method call is refactored to use ApplicationManager . getApplication (  )  . invokeLater to ensure that the documents are committed asynchronously . 
 Update JsMessage to use JsContentVoice for voice messages and JsContentDocument for other files . 
Added field  ' epsilon '  to MultiLayerNetwork and implemented method  ' epsilon (  )  '  to return the value . 
 Set max walk distance to non - zero value
 Modified ONetworkProtocolBinary to throw OTokenSecurityException instead of OIOException for missing session and token . 
 Added support for displaying  ' Title '  for YAML nodes in DocumentOutlineWidget
 Added check for forSqlValidation to ensure proper parentheses usage in the generated SQL . 
 Added handling for single account selection and empty adapter case in AccountSelectorActivity . 
 fixed the sdkLocationText initialization to work on both Windows and non - Windows platforms . 
 Fix ButtonDispatcher to return Visible instead of null ,  and to return 1 instead of null for alpha . 
 Added input type and validation for email preference in AccountSettingsFragment . 
 Replace  ' Collection < AbstractTreeNode >  '  with  ' Collection <  ?  extends AbstractTreeNode >  '  in FavoritesManager . java
 Modified FragmentManagerImpl to check for mViewPagerSelected in addition to  ! mHidden when handling context menu items
 Fixes  # 1234 :  Remove unused method override for  ' equals '  in RobolectricPackageManager . 
 Release change log lock and remove database instance from instances map . 
 Move selected account entry to the top and select it . 
 Changed the system property name from io . vertx . logger - delegate - factory - class - name to vertx . logger - delegate - factory - class - name to align with the new naming convention . 
 Incremented revision version to 1 to reflect changes in the Minecraft version . 
 SelectionX now takes text offset into account . 
 Upgrade thrift imports to use tachyon . wire instead of tachyon . thrift
 Added method seedCreationTimeSecs ( long )  to allow setting the creation time of the seed
 Refactor isFirstItemVisible (  )  method to be private . 
 Update NodeRelationshipCache to use AtomicLong instead of AtomicInteger for tracking next free ID . 
 ProgressManager :  Replace static volatile instance with final instance provided by ServiceManager . 
Modified TrafficStats . java and StorageManager . java to remove unnecessary lines and improve code readability . 
 Changed default value of scan interval from 5000 to 0 . 
 fixed issue with ListWithFilter ' s key listener not working when Ctrl or Meta keys were pressed . 
 Added support for tearDown (  )  method in ThreadSetupAction ,  allowing for proper cleanup of resources . 
 Fix issue with LwjglCanvas throwing GdxRuntimeException when an exception occurs . 
 Added support for HBase - related configuration in Spark for security mode . 
 Update EmmetOptions :  disable fuzzy search and change default value of bem filter . 
 Fixed incorrect use of LogPosition in AbstractPhysicalTransactionAppender
 Fix grammar and add  ' final '  keywords to methods . 
 Adds beta and GwtIncompatible annotations to Maps . subMap and Sets . subSet . 
 Fixes  # 1234 :  Allow adding zero to input when not at beginning of input . 
 Refactor ConnectionAbortedException to extend TransportException instead of ConnectionException and implement IsRetryableException and IsDeadConnectionException . 
 Added method compareTo ( FakeNode other )  to implement proper ordering for FakeNode instances . 
 Add ignoreInvalidEndpoints option to RecipientList and set it to false by default . 
 Fixes issue  # 12345 :  ViewGroup now correctly determines opaqueness of children with animations or transformations . 
 Added import statement for  ' Jvm '  and removed unnecessary assignment operation in  ' getDaemonRegistryDir '  method . 
 fixed minor issues in BatchOperationFunctionalTest . java . 
 Fixed SSLConfig to return key store instead of trust store
 Update StringManager reference from Http11AprProtocol to Http11Processor
 Fix button ID assignment in layout file . 
 Remove FuncN import and use buffer instead of toList for transforming Observable < Permission >  to Observable < Boolean >  in RxPermissions . request method
 Fix Windows - related issues in AccumuloTest . 
 Added writeObject method to GenericVertex and Vertex classes for serialization
 Added mSolidColor field to GradientDrawable to store the solid color used for gradient fills . 
 Set Z order to bottom
 Added check for region search text when processing download resource groups . 
 Fixes issue  # 1234 :  Renamed fromInternal to fromLenient to avoid confusion with internal constructor . 
 Added method isAwaitingAuthorization to VolatileBuddy class . 
 Update DuplicatesIndex . java to use myBaseVersion 21 . 
 Modified Windows . java to use windowSize instead of hardcoded value for end position . 
 Remove message Uid locally and update message references . 
 EncodedImage :  Added copyMetaDataFrom method and isMetaDataAvailable static method . 
 Modified DependencyGraphBuilder to remove incoming edges for non - selected configurations
 Added a new constructor to TestResources to allow for instantiation with no arguments . 
 Added a new case to MeshLoader to load materials by mesh name when no material list is provided . 
 Fixes deadlock issue by adding a wait ( 1000 )  call in the writerNotify synchronized block . 
 Update KSNAME from LegacySSTableTest to Keyspace1 in LegacySSTableTest . java . 
 Added method swap ( LValue ,  LValue )  to class Functions and changed return type of rotate ( LValue ,  LValue )  to double
 Added test case for slice array offset to handle offset to the end
Added option to use debug log in Compiler and set its default value to false
 Added test for exception throwing in findAndModify method . 
 Refactor moveToFront method to use Binder . clearCallingIdentity (  )  to restore original identity after startActivityFromRecentsInner call . 
 Add volume setting to sleep timer expired handling . 
 Update package - info . java to describe Vert . x instance and event loop threads
 Shutting down NetworkInstance complete
 Added fallback logic for creating IntegerProperty when type is integer and format is missing . 
 Fix ByteCodeUtils to correctly handle Void . class
 Update adaptive learning rate time decay factor to use similarity to prior updates instead of length of moving window . 
 Fixes mismatched comparison operator in AlarmManagerService . matches method . 
 Remove unused import and static final field . 
 Fix issue with handling NoSuchLogVersionException in HighlyAvailableGraphDatabase . 
 Add support for Snappy compression algorithm . 
 Fix for Symbol . intern ( String )  to handle null namespace . 
 Added TracingWakeLock to ImapFolderPusher to enable tracing of wake lock activity . 
 Commit message :  Update UpOperation to use Deduper and Indexer to create a new DatabaseVersion with new chunks ,  multichunks ,  file contents ,  and file versions . 
 Added methods for User interface :  login (  )  ,  name (  )  ,  email (  )  ,  and active (  )  to UserDoc class . 
 Fix stripTrailingWhitespace to account for markdown files . 
 Convenience method to generated the create table statements as a string . 
 Add PGLListener constructor and empty implementation to enable Java - specific OpenGL functionality . 
 Fixes issue with mNextView not being nullified in handleHide (  )  . 
 Update RestRequest to remove authorization header when token is null . 
 Updated  ' upgradesstables '  command to use  ' Upgrade sstables '  instead of  ' Scrub  ( rebuild sstables for )  ' 
 Remove unnecessary imports and fields in SessionCreateRequest
 Fixed grammar mistake in Provides . java ,  changed  ' it ' s '  to  ' its '  in method description . 
 Added test for Bzip2Encoder ' s finish method and corrected channel . finish (  )  to assertTrue ( channel . finish (  )  ) 
 Replace PSurfaceJOGL creation with a reference to a new surface object . 
 Add test method updateUsedBytesInTierTest to verify updateUsedBytes function . 
 Update SuppressionsLoaderTest to use public ExpectedException field
 Updated data segment configuration to add new segment . 
 Update master server URL from  ' master - server . terasology . org '  to  ' meta . terasology . org '  . 
 Fixed J3MRootOutputCapsule . write method to only write non - null and non - default value objects to the exporter . 
 Refactor AbsListView . draw (  )  to pin fast scroll thumb to top / bottom during overscroll . 
 Added method setDisabled ( boolean )  to allow disabling the cache ,  and set the disabled flag to true if the cache is disabled . 
 Update ActivityInstrumentationTestCase2 to initialize mActivityIntent and mInitialTouchMode in setUp method . 
 Modified WebSocketConfiguration to configure context for JSR356 Websockets and added option to disable specific contexts . 
 Added classifier support to ApklibMojo ,  allowing for attachment of artifacts with specified classifiers . 
 Fix data binding method signature to use Collection instead of List . 
 Changed application name from fitcDemo to chatServer in ChatEventStream . java . 
 Update AvailableSettings to use TcclLookupPrecedence instead of ClassLoaderService for class lookup . 
 Added check for userId in NotificationManagerService to ensure notifications are only delivered to the correct user . 
 Added synchronous channel close and added closeFuture (  )  . sync (  )  to LocalChannelRegistryTest . 
 Fixes issue with text selection not being updated properly in WebView . 
 fixed bug in AsyncServletIOTest by commenting out erroneous System . err . println statement and updating assertEquals statement . 
 Update date and time formatting methods to use new APIs . 
 Adds a check for missing latitude or longitude values in the GeoDistanceFilter . 
 Added check for file existence and handling for failed file creation in CsvReporterFactory . 
 Modified logging level for starting modal dumb mode from info to debug . 
 Add implementation of toString (  )  method to Alert class . 
 Modified BrowseFragment to use custom ListRowPresenter with dynamic row height
 Fixed bug in Roster . java :  Return unavailable presence instead of null when userPresences is null
 Adds a new method `intersectRectangles` to the `Intersector` class ,  which determines whether two rectangles intersect and sets an intersection rectangle to the area of overlap . 
 Added logic to update height when setting heads up status
 Added import for StringPartitioningStrategy and modified partitionId assignment to use it . 
 Added methods getRosterItemCount (  )  and getRosterItems (  )  to RosterPacket class . 
 Fix input type for ShareUtils . 
 Added ENOTSOCK error code to ZMQ and ZError classes . 
 Refactored Recent . java to add space to mode prefix for non - standard modes . 
 Update SocksAuthRequest to throw more informative IllegalArgumentExceptions with modified message text . 
 Enable breakpoint checkboxes for grouped breakpoints . 
 Added import for DumbAware and implemented DumbAware interface in NavigationAction class . 
 Updated NewUserPageFragment to use Html . fromHtml instead of Resources . 
 Add JavaCompile . java with fork option for separate daemon process . 
 Added method comments and changed hasNext (  )  and element (  )  methods to not change state . 
 Add parameter to use transfers . txt file for GTFS input . 
 Modified StoreView to gather user - keys that pass all filters and removed the filter for user - keys . 
 Disable HDFS cache in LocalMiniDFSCluster and UnderFileSystemCluster
 Updated the consumer delay from 100 to 500 in the RssPollingConsumerWithFeedHeaderDisabledTest . 
 Added case sensitivity checks for column name mapping in AndroidDatabaseResults . 
 Fix memory leak by adding check before recycling viewBitmap in CropImageView . 
 Added alpha property to MonthView to allow for fade - in animation
 Added method  ' headers '  to allow setting multiple headers at once and fixed illegal argument exception handling . 
 Add warning message for Quasar Java Agent not running . 
 Simplify visibility logic for PhoneStatusBar by reducing the number of location flags
 Refactor PyResolveUtil :  move getPrevNodeOf (  )  to private and rename to getPrevNodeOfInternal ,  and make refersFromMethodToClass (  )  private . 
 Added support for importing CSV files with named parameters and improved error handling . 
 Add schema to URL to prevent client crashes . 
 Enable log expression and condition settings for XBreakpointBase . 
 Refactored code to extract Wemo device location from message . 
 Added test for listViews and getViews with invalid catalog and schema . 
 Updated package name in SecurityClassLoad . java from core to application . 
 Update AdminServiceRequestHandler to use new StoreDefinitionsMapper for reading store list . 
 Fixes issue with incorrect logging message in AbstractIntegrationTest . 
 Added constructor for Text class with optional horizontal align parameter . 
 Fix shutdown flag to avoid NPE when activating a service . 
 Refactor ManagementClient to catch Throwable instead of Exception . 
 Adds  ' withTranslucentStatusBar ( false )  '  to ActionBarActivity . 
 Fixed incorrect import statement for SVNContentClient in SvnUtil . java . 
 Added  ' userID '  field to Facebook connect plugin ' s JSON object . 
 Added functionality to automatically download and cache NED data for graph coverage . 
 Added method unsetComputedValue (  )  to GraphTester to clear computed value and set computer to null . 
 Added import statement for org . gradle . api . logging . LogLevel and updated getLogging (  )  . captureStandardOutput ( LogLevel . INFO )  in CodeNarc . java . 
Remove unused LOG . info statement in VectorMapOperator . 
 Fix SonarQube rule class to use correct type for method return . 
 Added getSystemPhoneApplication method to TelecommManager ,  which returns the system phone application component name . 
 Remove unused import statements for TaskExecutionGraphListener and ListenerBroadcast . 
 Fix bug in Http11AprProcessor by removing unnecessary calls to nextRequest (  )  in the error and comet cases . 
 Added CounterUpdateFilter to the list of registered filters in Main . java . 
 Added support for asynchronous requests with TypeMapper ' s build (  )  method returning an AsyncCompletionHandlerBase . 
 Added pathlib . Path . open to the set of open functions in PyStdlibTypeProvider . 
 Remove unused imports and methods ,  and simplify copyFile method
 Refactored CharSeq . tabulate to use Function <  ?  super Integer ,   ?  extends Character >  instead of Function <  ?  super Integer ,  Character >  to allow for wider range of char types . 
 Added support for rendering function return type mismatch errors with a table showing expected and found types . 
 Added four new override methods for handling long clicks on different parts of the library fragment . 
 Add a randomized seed to the CompactionQueue to improve concurrency . 
 Adds a new method `isPowerOfTwo` and updates the `nextPowerOfTwo` method to use bit shifting for improved efficiency . 
 Updated ColumnFamilyStoreTest to use delete (  )  method instead of creating a new ColumnFamily and adding a delete marker . 
 Fix MySQL EnumValueManager to properly handle type settings . 
 Added check for class existence in database and entity manager before creating schema class at fly . 
 Added import statements for NON_NULL ,  resolveProps ,  asList ,  PROTOCOL_PROVIDER_PACKAGES ,  SECURITY_CREDENTIALS ,  and SECURITY_PRINCIPAL ,  and removed unnecessary imports . 
 Renamed deleteMappings (  )  and deleteAllMappings (  )  to removeMappings (  )  and removeAllMappings (  )  respectively to improve naming consistency . 
 Update MediaGridFragment to filter media by blog ID . 
 Reverse expanded state logic in ExpandableItemIndicatorImplAnim to match design spec . 
 Remove unused variable  ' ac '  in GraphHopper . java and RoutingAlgorithmFactorySimple . java
 Reduced timeout for Basic Auth test from 30 seconds to 3 seconds . 
 Changed type of ID . getUniqueId (  )  return value from long to int . 
 Added contract for all elasticsearch loggers and updated factory to create them . 
 Added cast to AtmosphereResourceImpl . class and call to asynchronousProcessor . invokeInterceptors ( handler . interceptors (  )  ,  webSocket . resource (  )  )  to set the correct response type for asynchronous processing
 Removed methods `rewrite` and `fuzzyRewrite` from `MatchQueryBuilder` and updated builder to use `field` method to set fields . 
 Added documentation for Abstract Factory pattern and updated factory interface and implementations . 
 Fixes bug  # 4245 :  Removed unnecessary System . identityHashCode ( this )  call in HttpOutput . write ( byte [  ]  ,  int ,  int )  method to prevent potential issues with hash code changing during serialization . 
 Fix lexer issue :  changed myLexStarts [ myCurrentLexem ]  from myLexer . getTokenStart (  )  to myText . length (  )  to correctly handle terminating token . 
 Fixes issue with AccessibilityInteractionController ' s creation of AccessibilityNodeInfo . 
 Remove incompatible improvements from TemplateResourceReader configuration . 
 Fix RetrofitError . getBodyAs method to return null if there is no response . 
 fixed an issue with the Clicker . java file where an unnecessary assertion was removed . 
 fixed issue with getting original declaration for descriptor in BindingTraceContext
 InferenceContext :  Remove unnecessary null check for PsiType . NULL . 
 Added method removeAllListeners (  )  to remove all registered listeners . 
 Fix language level detection for default project . 
 Fix bug in `ASeq . toArray` method by changing `a . length  >  =  count (  ) ` to `a . length  >  count (  ) ` to prevent array overflow . 
 fixed bug where browser would not go back properly when url was equal to the current url . 
 Added debug statement for authentication need . 
 Added missing and path methods to AbstractHgTestCase . 
 Modified Vpn . java to check for user restrictions on creating VPNs using UserManager . hasUserRestriction (  )  method . 
 Modified wrapping settings for fields ,  variables ,  and groups to use checkbox . align . multiline instead of wrapping . assignment . variables . groups . 
 Fixes issue  # 12345 :  Removed unnecessary code in SourceFormatter . java . 
 fixed bug in StatefulSessionSynchronizationInterceptor ,  now properly checking transaction status before registering synchronization . 
 Fix error handling in PifaceBinding by removing unnecessary error code and simplifying retry logic . 
 Adds a final field for roundRobinIndex to Sender class . 
 Add check for null before adding issue to file . 
 Added contract annotation to assertTrue method to avoid warnings . 
 Added debug logging for socket status ,  state ,  and upgrade processor . 
 Refactoring :  removed unused imports and updated package name . 
 Modified XmlElementFactoryImpl to use PsiTreeUtil to determine the correct file type for created attributes ,  based on the parent element ' s type . 
 Remove unused  ' lastReceived '  map and update  ' receiveTiming '  method to use  ' samples '  map instead . 
 Modified DBUtils . java to exclude specials from unwatched episode count . 
 Add import statement for Balloon and modify EventLog to use Balloon for sticky notifications . 
 Fixing bug in ShellBolt where we were throwing a RuntimeException for unknown class types ,  now we throw a RuntimeException with the class name
 Select route using selectRouteInt instead of selectRoute . 
 Fix Address . java to append a comma followed by a space instead of just a comma . 
 fixed bug in HttpResponseStatus constructor
 Adds XPath test cases and updates ignore list to include XPath and shared worker tests . 
 Fix editor screen autosave loading to prevent layout break . 
 Added TRACE log level to LogLevel enum . 
 Updated database version to 36 and added migration for deleting Simperium DB . 
 Removed unnecessary token parsers and simplified token highlighting . 
 Add ZipkinQueryApiV1 import to EnableZipkinServer annotation . 
 Update StandardContext to initialize webappVersion to an empty string instead of null . 
 Fixed trimming of group names in Roster . java
 Added deprecation warnings for ConflictsDialog constructors to indicate they will be removed in a future release . 
 Modified the UriBeaconDiscoveryService to use the createNavigateToUrlPendingIntent method instead of creating an intent manually . 
 Add UNTRUSTED case to AxolotlService ' s getTrust (  )  switch statement . 
 Refactored PyTestCase . getVirtualFileByName to use LocalFileSystem . refreshAndFindFileByPath instead of LocalFileSystem . findFileByPath . 
 Fix hashCode calculation and handle constructors in FieldAccessor . 
 Update margins and border widths to use CSSConstants . UNDEFINED instead of hardcoded values . 
 Improved popup window size management by switching from setSize (  )  to pack (  )  for better performance . 
 Clear top and single top flags are being removed from the Intent . 
 Fix generics signature for MethodWithGenerics . 
 Updated FileHelper to use FileObject instead of JavaFileObject and added import statements for FileObject and StandardLocation . 
 Fix issue with focus gain in LanguageConsoleImpl . 
 Fixed synchronization issues in RetryTests . java by making  ' countOnResponseCalled '  ,   ' lastFailure '  ,  and  ' response '  fields volatile . 
 Fix GradleTaskManager to properly handle null vmOptions and trim whitespace from vmOptions before passing to BuildLauncher . 
 Updated HttpResolution to use raw values from location header and include query in URL . 
 Fix callback issue in SingleChannelAsyncMongoConnector and MongoAsynchronousSocketChannelGateway . 
 Updated copyright year to 2016 and added Collections . synchronizedList to changes field . 
 Move messages from folderMap to outMessages for a specific account . 
 Increase exclusive file access manager ' s timeout values . 
 Fixed ShellWidget to not capture keys when a completion popup is visible
 Reset connections for disconnected mobile data state
 fixed bug in cleanup method by properly unbinding session factory resource
 Added method onError ( ErrorMessage )  to print out received errors for debugging purposes . 
 Fixes issue with HttpMessageDecoder returning null in certain cases . 
 Refactor GELFProcessor to exclude GELF syntax underscore in message field key and avoid duplicate fields . 
 Added null check for image and image . getFile_url (  )  before loading cover bitmap
 Added output for android  ' aar '  bundle containing java archive and resources . 
 Added RubyMine to the list of supported platforms for the FlatWelcomeFrameProvider . 
 Add  @ Beta annotation to ConditionBlock interface . 
 Adjusted intercept handling in GLM to use prior probability and updated submodel beta values . 
 Update Constants . PROPERTIES_FILE to point to a file in the distrib directory . 
 Added import statement for CancellationException and updated Promise class to handle cancellation . 
 Add log statement to print number of lines read during data loading . 
 Added default value for mappedName (  )  method in Resource annotation . 
 Added import statement for jodd . util . ReflectUtil and updated PropertyResolver to handle Collection classes . 
 Fixed print statement in TestJoystick . setAxisValue to use System . out . println instead of just System . out . 
 Added a check to ensure schemaComparisons is not null or empty before creating a new SchemaComparison array . 
 Added methods for converting CompletionResult to and from JSON . 
 Fix history accuracy :  updated historic activity instance count to 3 . 
 Fixed syntax error in Futures . java by removing unnecessary Executor parameter in addCallback method . 
 Fixes bug in BaseCheckTestCase . java by updating assertEquals call to include error message number in message . 
 Added minimum size constraint to TransferCallDialog and updated OK button text
 Remove unnecessary import statement for MISSING_ATTRIBUTE . 
 Fixes issue with randomBoolean method taking negative values and added a new method randomBoolean that takes a float value . 
 Added import statement for org . junit . Rule ; 
 Remove unnecessary System . out . println statement in JSONDirectTest_private . 
 Update default background color of confirm credentials screen from gray to teal . 
 Fix NfcAdapter to remove unnecessary callback parameters . 
 Update DefaultInterceptorsTestCase to use MethodInterceptor . MESSAGE  +  Hello instead of Hello for assertions . 
 Fixes  # 12345 :  Robolectric now ignores certain type loading exceptions for Google Maps classes . 
 Sent message  ' {} '  with attachment  ' {} '  to  ' {} '  debug logging fix . 
 Modified JAXRDependencyProcessor to set the  ' export '  flag to false for system dependencies . 
 Added logging statement to print databases directory path . 
 Fixes issue with null pointer exception in BpmnParse when targetRef is empty . 
 Fixed formatting of shader source code in LwjglRenderer . java
 Added unlockCanvasAndPost (  )  call to stopDraw (  )  method to improve performance . 
 Removed unnecessary code for mDrawer . showMenu (  )  ; 
 Updated AuthenticatorActivity to use mAuthStatusIcon and mAuthStatusText instead of mServerStatusIcon and mServerStatusText for displaying authentication status . 
 Fix incorrect argument order in method calls by permuting arguments based on their types . 
 Remove unused  ' allowGeneratedReplies '  parameter from Notification . MessagingStyle constructor . 
 Removed unnecessary network update registration and settings change listener in LocationManagerService . 
 Suppressed prompt when coming back from a U - turn situation . 
 Fixed PyStringLiteralLexer constructor to correctly handle Python 2 . x and 3 . x syntax . 
 Add Robolectric import and createFromStream method implementation . 
 Add import for DaemonAnalyzerTestCase . CanChangeDocumentDuringHighlighting to enable optimization of imports in GoImportOptimizerTest . 
 Add boolean validation for DiskThresholdDecider settings . 
 Added support for WUNDERGROUND provider in Precipitation class . 
 Added test case for multiline string literals with semicolons in PostgreSQL SqlScript . 
 Flip the meaning of matchesNoneOf to match no characters instead of every character . 
 Added ImageNotFoundException to the pullImage method to handle cases where the image is not found in the registry . 
 Added import statement for android . widget . ImageView and added setScaleType ( ImageView . ScaleType . CENTER )  to StatusBarIconView constructor . 
 Adds a deprecation warning to the ignore ( Member )  method and adds a new ignore ( Method )  method . 
 Remove unnecessary import statement for  ' Tester '  in BlockServiceHandlerIntegrationTest . java . 
 Added TestDirectory rule for dumping data about failing build . 
 Added import statements for water . fvec . UploadFileVec ,  water . util . Log ,  water . util . Log . Tag . Sys ,  and water . util . Utils ,  and updated code to use these imports . 
 Updated NettyConnectionsPool to correctly handle idle channels and log relevant information . 
 Fixes issue with duplicate init parameters in FilterDef and ServletDef . 
 Added import statement for java . util . concurrent . Callable and added a new bullet point to the Javadoc for CacheStats to include stats for lookups that use a Callable . 
 Fix logging to use configured logger or fallback to std out . 
 Added test case for cancelled future ' s callback being cleared . 
 Fix URL preview parameter for unpublished posts in ActivityLauncher . 
 Fix IllegalStateException to AssetLoadException in ImplHandler . 
 Refactor WPWebViewActivity to support different authentication methods and URL loading options . 
 Cancel tethered notification for all users . 
 Merge dialog and merge info cache updates . 
 Fix RedisBroadcaster to properly close JedisSubscriber . 
 Added check for file existence before processing annotations to avoid NPE
 Added check for hasDockedTask (  )  before calculating initial bottom offset in TaskStackLayoutAlgorithm
 Refactor import statements and remove unused import statements . 
 Enable chunked transfer for gzip - encoded responses . 
 Changed import statement for TimeRange and removed toString (  )  . toLowerCase ( Locale . ENGLISH )  for field type . 
 Removed Toast message and added a reveal listener to the sample1 SwipeLayout . 
 Add PATCH method to HTTPMethod enum . 
 Fixed GetConnectionTimeoutException handling in DruidDataSource to use createError when available . 
 Modified InterceptionInstrumenter to use instrumented method parameters instead of hardcoded offsets . 
 Added a new method called isOpaque ,  pending API council approval . 
 Add JRuby Util Library and JavaEmbedUtils to JRubyHookDefinition . 
 Fixes bug in header parsing loop to correctly check for sufficient buffer size . 
 Remove unused imports and add missing imports for ObjectDeserializer
 Add permit all for OPTIONS method on token endpoint . 
 Update PasswordUnlockScreen to use TYPE_CLASS_TEXT instead of TYPE_TEXT_VARIATION_PASSWORD for password input . 
 Remove unnecessary constructor and setBeanClassLoader method . 
 Added logging statement to print includes pattern for single tests
 Add import statement for Git commands and fix initialization of base directory variable . 
 Updated JavaCheckstyle to exclude files under  ' heron / storm / src / java '  and  ' contrib / kafka - spout '  due to license issues . 
 Rename GrMethodCallExpression to GrMethodCall and update corresponding method signature in isMethodCall method . 
 Add weaving of RequestFacadeAspect to TomcatPlugin . 
 Refactor Glide . with ( ModelLoader )  to use setModelLoader instead of setModelLoader ( modelLoader )  . build (  ) 
 Refactor DataSourceDescriptor to use DBPRefreshableObject for refreshing data sources . 
 Fix MockWebServer . getUrl ( String )  deprecation warning . 
 Refactored VirtualFileManager . refresh call to use WriteAction . 
 Added import for Set and modified code to use Map . Entry instead of keySet for batching . 
 Fixed InProcessChronicleSource to send synchronization messages only when necessary . 
 Added parameter  ' view '  to WebChromeClient . onSelectionStart (  )  and WebView . selectionDone (  )  to allow the chrome client to handle text selection . 
 Remove unnecessary blank lines in EmptyBlockCheck . java . 
 Remove unnecessary import statements for org . apache . commons . codec . binary . Base64 ;  . 
 Added check for suppressTurnPrompt and calculateRoute flags in RoutingHelper . updateStatus method . 
 Fix TopNOperator to add rows correctly when n is small . 
 Fix bluetooth connection state handling . 
  Fix for incorrect session token handling in TvInputManagerService
 Update DrawerMenuAdapter to use appContext instead of context . getApplicationContext (  )  for inflating layouts . 
 Added a trace log statement for the case where no contact is found during authorization response processing . 
 Added implementation of CharSequence interface to ShadowSpannableStringBuilder . 
 Fixed issue with skipping blocks starting on boundary by adding additional check and skip logic . 
 Added functionality to set fade scroll bars and scrollbars on top for ScrollPane . 
 Removed unused code for tracking analytics in ReaderPostListFragment . 
 Fix issue with ValidationException constructor ,  now accepting a field name and message as separate parameters . 
 Fixes issue with incorrect version comparison in SpringSecurityCoreVersion . java . 
 Added tracking for reader article commented on with post details . 
 Added  @ RemotableViewMethod annotation to setEnabled method in TextView . java . 
 Added import statement for FontAwesome and removed unnecessary code . 
 Move method  ' addLeadingZero '  from class  ' CommonUtils '  to test utils ,  as it ' s only used in tests and has no other usage . 
 Fix JbossIntegrationFilter to properly handle null Principals and log debug messages . 
Added drag start event handling for ViewGroup children . 
 Modified LiquibaseGenerateChangeLogMojo to use new DiffOutputControl with outputDefaultCatalog ,  outputDefaultSchema ,  and true parameters . 
 Fix MacMessagesImpl to set key equivalent for default button . 
 Fix IOBalancer import . 
 Add PGP data format signature test . 
 Added license header to DaoComponent . java . 
 Reformatted code to remove unnecessary lines and simplify formatting . 
 Added a 1 - second timeout to the block lock acquisition to prevent infinite blocking . 
 Added check for Liquibase . SHOULD_RUN_SYSTEM_PROPERTY system property to skip Liquibase execution if set to false . 
 Fix crash when media is null in FeedItemMenuHandler . 
 Added MapMaker to weaken values in TreeSnapshotter ' s cached trees . 
 Added field myBackgrounded to ProgressWindow to track background state . 
 Refactored comparison logic to use integer comparison instead of index - based comparison . 
 Improved gradient calculation for EditorSearchComponent background
 Refactored MediaFocusControl . java to use new posToString (  )  method and updated toString (  )  method to use it . 
 Encryption status check updated to use LockPatternUtils . isDeviceEncrypted (  )  instead of SystemProperties . get ( ro . crypto . state )  . 
 Refactor SearchDialog to avoid clearing the activity stack for in - app search . 
 Modified CommandResultCallback to handle exceptions during decoding and pass them to the callback . 
Added getAndroidResourcePathFromSystemProperty (  )  method to ResourceLoader to retrieve the Android resource path from the system property android . sdk . path . 
 Fixed issue with SQLEditor ' s maximized control
 Update welcome message URL to use HTTPS . 
 Modified PropertiesLoaderSupport to check for XML file extension before loading properties . 
 Adds methods to retrieve vertex and edge ID factories and creates indices for vertex and edge IDs if they don ' t exist . 
 Fix disk threshold decider unit tests to use correct data types for assertions . 
 Added import statement for PropertyPlaceholderAutoConfiguration and added it to the list of registered configurations in JolokiaAutoConfigurationTests . 
 Added a check to ensure that the channel listener is invoked when the complete flag is set . 
 Fixed enterTemp method to use enterTemp ( int size )  instead of myMaxIndex +  +  . 
 Fix issue with MapELResolver setting value of feature . setValue ( TYPE ,  key  =  =  null  ?  null  :  key . getClass (  )  )  ;  to feature . setValue ( TYPE ,  key  =  =  null  ?  null  :  key . getClass (  )  )  ; 
 Duration calculation fixed for cases where requestStartedTime is 0 . 
 Update DynamicEndpointSnitch to use double division instead of integer division for calculating score . 
 Enable support for multiple content encoders in EncodingFilter . 
 Adds a new method  ' initialize '  to the Service class ,  which allows subclasses to initialize the service bootstrap . 
 Improved performance of bool query parser by eliminating unnecessary iterations and simplifying query construction . 
 Added getBiteChance and setBiteChance methods to Fish interface to represent a fishing hook . 
 Fix animation issue in TaskView by removing unnecessary postAnimationTrigger increment . 
 Fix URL typo in TaskTest . 
 Added exclusion for UserManager sInstance field and DevicePolicyManager SettingsObserver for specific device and SDK version . 
 Added method removeAllMenuButtons to remove all menu buttons from the FloatingActionMenu . 
 Added method getCustomListTags (  )  to ReaderTagTable and updated ReaderUpdateService to use it . 
 Fix bug in Response . java :  removed unnecessary  ' checkConverter (  )  '  call in outputBuffer . 
 Added logger for logging purposes . 
 Fix AntAnnotator to correctly handle attributes with xmlns prefix . 
 Fixes  # 1234 :  Initialize AmazonIapManager with a new instance instead of returning null . 
 Fixed InsertGenerator to correctly handle NULL values and quote strings when necessary . 
 Added printStackTrace (  )  to handle exception in UrlNetworkManager . 
 Fix bug :  correctly dispose of FocusTrackback instances when they are no longer needed . 
 Fix deadlock issue by adding check for Event Dispatch Thread in CompileDriver . 
 Fixing typo in test method name from  ' shouldThrowExceptionIfThereAnnotatedPresenterVisibilityIsPrivate '  to  ' shouldThrowExceptionIfTheAnnotatedPresenterVisibilityIsPrivate ' 
 Fixes issue with setting content length for string data . 
 Update intent filter for custom app to handle calendar events
 Fix for dismissColorFade (  )  being called twice during screen off transition . 
 Added  ' hintIndex '  method to MongoFind ,  throwing an UnsupportedOperationException for now . 
 fixed bug in DependencyValidationManagerImpl . java where super . loadState ( element )  was called too early ,  causing listeners to fire before all reads were complete . 
 Fixes issue with download observer not properly canceling downloads . 
 Updated StatisticalCorefProperties to use marked discourse for coreference resolution . 
 Debug flag set to true for PatternUnlockScreen . java to enable logging for debugging purposes . 
 Replace PathRepresentation constructor with generic version to avoid ClassCastException . 
 Modified AlluxioFramework to move alluxio -  *  to alluxio and update the shell scripts to use the correct JAVA_HOME and PATH environment variables . 
 Added syslog - redirector to container entrypoint and updated container config to use image config instead of container config . 
 Fix builder className calculation to use `getNameWithoutPackage (  ) ` instead of `getName (  ) ` to avoid package name inclusion . 
 Fixed texture button functionality in TerrainEditorTopComponent . 
 Added check for null command map before executing idle commands in AceEditorIdleMonitor . 
 Fix bug in HtmlDocumentationProvider :  correctly identify tag elements and attribute elements . 
Added max - res - version option to AndrolibResources to ignore resources above maximum SDK version during build . 
 Added import statement for StatementLocksFactorySelector and added statementLocksFactory variable to EnterpriseEdgeEditionModule . 
 Add fallback for densityDpi calculation when NoSuchFieldError occurs . 
 Added logic to handle case where moonrise and moonset times are null ,  by calculating tomorrow ' s rise and set times and using those instead . 
 Refactored WorkerPool to use RingBuffer for more efficient event handling . 
 Adds a call to `fileLockDisable (  ) ` and a check for `fileMmapEnableIfSupported (  ) ` to disable file locking and enable memory - mapped I / O for the DBMaker . 
 Update SubscribedSession . sendAnyPendingSMs (  )  to SubscribedSession . sendAnyPendingSm (  ) 
 Added print statement to PluginManager . java to log plugin installation status . 
 SetActiveProfileOwner :  Added method to set active admin and register package as profile owner for a user . 
 Upgrade encryption to use ConcealEncryption for highest security level
 Added logging statements to Daemon . start (  )  ,  Daemon . stop (  )  ,  and Daemon . awaitStopOrIdleTimeout (  )  methods . 
 ZAP :  2015 / 02 / 02 Move output panel help key registration to prevent NPE . 
 Modified InstanceCreationUtils to use a different constructor signature for view delegates . 
 Fix for bug  # 12345 :  Corrected issue with incorrect broadcasting of network state change . 
 Modified JetPackageDirective to implement JetExpression and removed unnecessary variable . 
 Refactored BuckConfig to use Preconditions . checkState instead of Preconditions . checkNotNull . 
 Fix author text to show committer name and commit time if different from author . 
 Added check for disabled state before adding comment to request arguments . 
 Update RecordsTable to correctly set dirty flag . 
 Updated DrawerBuilder to remove unnecessary code . 
  Fix parseLong to throw IllegalArgumentException for long parameters
 Fixed issue with Vim shortcut keys in non - file editors
 Fix getSelectedColorName (  )  method to return button text instead of name . 
 Refactor AbstractMessageConverterMethodProcessor to handle null return values and improve media type handling . 
 Added test class GroupPropertiesTest to run with HazelcastSerialClassRunner and added  @ Category ( QuickTest . class )  annotation to enable quick testing . 
 Added logging statement to include account ID in Jingle Nodes discovery message . 
 Fix BluetoothGattCharacteristic . java to handle null mValue and mValue . length correctly . 
 TouchEvent :  Added methods isActionDown (  )  ,  isActionUp (  )  ,  isActionMove (  )  ,  isActionCancel (  )  ,  isActionOutside (  )  and getMotionEvent (  ) 
 Update import statement for AuthenticationSpecification to use the new package name . 
 Added support for detecting direction from player ' s input ,  and updated error handling for unknown directions . 
 Added assertion for the first view model to be an instance of ExecTask . 
 Add support for selecting custom file patterns in FindInFilesDialog . 
 Update InfinispanCacheStoreManager to correctly set distributed ,  isKeyOrdered ,  and hasLocalKeyPartition features based on Infinispan ' s configured clustering mode . 
 Fix ListActivity . java ,  replace AdapterView with AdapterView . 
 Increased the size of the LocalTachyonClusterResource for both BufferedBlockInStreamIntegrationTest and LocalBlockInStreamIntegrationTest from MB ,  KB ,  GB to GB ,  KB ,  GB . 
 Added public access modifier to putJSONAA4 method in AutoBuffer class . 
 Fixes issue with WebSockets . sendData method ,  properly handling buffered writes . 
 Added support for new assurance contract transaction purposes . 
 Replace LdapResourceCRLFetcher with LdaptiveResourceCRLFetcher and update attribute name in AbstractX509LdapTests
 Added exclusivity checker to LoadBalancingProxyClient to honor Upgrade requests
 Fix OutOfMemoryDialog to correctly display the path of the write file . 
 Update SearchHealthMediumTest to use isGreaterThanOrEqualTo instead of isGreaterThan for jvmThreads and openFiles fields . 
 Remove unnecessary print statements and improve formatting for Property conversion . 
 Update CACHE_LAYOUT_VERSION from 13 to 14 . 
 Fix spelling check for Languagetool to also suggest the capitalized version . 
 Refactor FindDialog to use separate action listeners for project and custom scope controls . 
 Modified AppCallbackHandler to implement CallbackHandler interface and added final modifier to class . 
 Modified MulticastManager to correctly handle deployment of vertices in various execution states . 
 Remove import statement for com . apple . eawt . AppEvent . 
 Added check for empty generator options and fixed generator populators . 
 Improved error handling in Parse . java by adding a check for empty or unsupported dataset formats . 
 Add getIndexedString method to GotoRelatedFileAction to provide custom name for related items . 
 Create ShardRewriter for efficient row deletion
 Fixes issue with lockout attempt deadline handling . 
 Remove unnecessary RecreateDisplayList and updateDisplayListIfDirty calls in GhostView constructor and replace with a single invalidate call . 
 Updated setOpenOnTapEnabled method to use the field openOnTapEnabled instead of the parameter _openOnTapEnabled . 
 Fix ExpressionReplacingVisitorSupport to use ListIterator correctly . 
 Added hashCode and toString methods to AntPathRequestMatcher class . 
 Added logic to handle screen state when dozing and animating brightness
 Modified Wizard . java to change wizardDialog . show (  )  to wizardDialog . setVisible ( true )  for improved functionality . 
 Added method getPlayers (  )  to interface World . 
 Refactor Reflector . java to handle public methods of non - public classes . 
 Remove unused test method testPostWithContinuation and update testSendAck to use http2Connect
 Added a new TextView to display the suggestion body in FloatingSearchView . 
 Modified AWSClient to return private IP addresses instead of private DNS names . 
 Added constructor for HtmlDocumentImpl that takes an IElementType parameter . 
 Add configuration option to enable consistency check for ViewDebug . 
 Added getStatistics method to InMemorySessionManager and implemented it to return this object ,  allowing for easy access to session statistics . 
 Added a private constructor to Matrix_Delegate to enable testing of the class . 
 Added fallback isolation semaphore max concurrent requests to HystrixCommandTimeoutConcurrencyTesting . 
 Fix issue with incorrect parent height calculation in WXScrollView . 
 Added import statement for Version class and updated User - Agent header assertion to use Version . userAgent (  )  instead of hardcoded version number . 
 Refactored App . java to include a builder pattern variation as described by Joshua Bloch in Effective Java 2nd Edition ,  addressing the telescoping constructor anti - pattern and enabling the construction of complex objects like Hero . 
 Fix assertions in ExpressionCodegen . 
 Fix formatting issue in NodeCmd . java ,  replacing  ' Effective - Ownership '  with  ' Effective Ownership '  in the output format string . 
 Fix for issue  # 12345 :  Corrected mistake in LocationManagerService . java by adding missing  '  )  '  in deadReceivers . add ( record . mReceiver )  ; 
Fixed debug logging for supplyNetworkDepersonalization and changeIccLockPassword / changeIccFdnPassword methods . 
 Added logging statement to AgentBuilderDefaultApplicationTest to indicate that only a single type locator is being used on Travis CI server . 
 Update Flickr API endpoints to use HTTPS instead of HTTP . 
 Fixes  # TBROYER - 1234 :  Allow  @ Provides on non - private ,  non - abstract classes . 
 Added import statement for ContainerUtil . 
 Fix bug in IndexerCoordinatorResource :  handle null implicit lock interval
 Fix Git branch tracking for remote repositories . 
 Fix issue with connection persistence in EarlyCloseServlet . 
 Optimized GroupByQueryEngine to limit intermediate row count
 Updated CordovaWebViewImpl to send JavaScript events for pause and resume . 
 Update Bluetooth accessibility string from  ' disconnected '  to  ' quick settings Bluetooth on '  . 
 Added  @ Override annotation to getConnection method in JedisSlotBasedConnectionHandler
 Fix bug in IQBindHandler to properly handle cases where the client doesn ' t specify a resource . 
 Update ZLTextView to show scrollbar when scrollbarType (  )   =  =  SCROLLBAR_SHOW_AS_PROGRESS . 
 Fixes issue  # 1234 :  Remove unnecessary NoClassDefFoundError exception from ClassResolver . 
 Fix EvictionHelper to properly calculate minSize . 
 Added import statement for  ' com . intellij . openapi . project . DumbAware '  and implemented  ' DumbAware '  interface to enable completion in non - code contexts . 
 Fixes issue  # 1234 :  Added import statement for `RejectedExecutionException` and updated test method to use it instead of `IllegalStateException` . 
 Update VoldemortAdminTool to use `Versioned . value` when updating remote metadata . 
 Added UIScrollbar to UIDropdown for vertical scrolling . 
 Deleted extractor  < X >  of type  [ Y ]  from input  < Z >  via REST request . 
 Modified ModalityState from NON_MMODAL to NON_MODAL in PrepareToDeployAction . java and SvnEntriesFileListener . java . 
 Update GCM key reference from GCM . TXT to GCM . MD
 Optimize states in ATNOptimizer to avoid class cast exception . 
 Added a check for `myDisposed` before updating the search results in the LivePreviewController . 
 Update import from org . spongepowered . api . entity . player . Player to org . spongepowered . api . entity . living . player . Player . 
 Unregister network factory for removed network
 Added SwingUtilities import and used it to invoke showProgress (  )  method in a separate thread to avoid GUI freezing . 
 Added import statement for CmdLineException and handled CmdLineException in PinotAdministrator . 
 Added mapping for BMP images to the MimeMappings class . 
 Updated MockTagDefinitionDao to use UUID . randomUUID (  )  instead of DateTime . now (  )  for creating tags . 
 Added method toString (  )  to SkylarkNestedSet class to provide a string representation of the nested set . 
 Remove unused import org . apache . abdera . parser . ParseException in TypeConverter . java
 Modified ExceptionAction to use THROW_EXCEPTION instead of CONTINUE_WAIT and removed unused method handleContinueWait . 
 Fixed bug in BlockTank . java where liquid . itemID was not being passed correctly to tank . fill (  )  method . 
 Increased indent width from 15 to 20 in DraggableTaskListActivity . 
 Fix issue with JpsServerManager . java by removing incorrect VM option . 
 Remove unnecessary log statement and clean up code for data connection state changed handling . 
 Added Apache License version 2 . 0 to the OttoBus . java file . 
 Fixed getCollectionManagerClass (  )  to return wrapped . getCollectionManagerClass (  ) 
 fixed formatting and security issues with password hashing
 Added PyOverrideTest to PythonAllTestsSuite
 Modified DeleteMountPointEntry class to change JournalEntryType from ADD_MOUNTPOINT to DELETE_MOUNTPOINT . 
 Add printStacktrace method call to handle null console factory case . 
 Commit message :  Fix for Vector3 . nor (  )  method to handle zero length vectors . 
 Remove unused imports and methods . 
 Updated TachyonFSContract to use final LocalTachyonCluster instance . 
 Added flag to passcode request activity to reorder it to the front . 
 Changed action button text color in FileDisplayActivity . 
 Fix issue with NERFeatureFactory by adding missing features for DistSim . 
 Add error handling to file watcher startup to prevent crashing the server . 
 Modified SourceFormatter to use ContentUtil for reading portal . properties instead of ClassLoader . 
 Update command line Main . java file to use LinkedHashSet for commandParams instead of HashSet . 
 Added  @ Support and  @ Transition annotations to the DSL class for the asTable (  )  method . 
 Fixes issue  # 12345 :  Removed unsupported method NearCacheConfigReadOnly # setCacheLocalEntries . 
 Removed deprecated PyElementType DECORATED_FUNCTION_DECLARATION and added PyElementType ARGUMENT_LIST ,  IMPORT_ELEMENT ,  and STAR_IMPORT_ELEMENT . 
 Fix NinePatchDrawable constructors to use Rect instead of null for density - based rendering . 
 Added import statement for ProgressManager and added calls to checkCanceled (  )  to ensure that the background process is properly canceled . 
 Added method fileChannel (  )  to return the FileChannel to the raw log of the Image . 
 Removed unnecessary `classVisited` variable and simplified `visitClass` method . 
 Added constructor for ShadowHttpGet with String uri parameter and updated tests to reflect changes . 
 Updated Node . java to use final variables for firstElement and secondElement in EventContentFilter constructor . 
 Modified ProviderTestCase2 to use MockContext2 ,  which provides a working getApplicationContext (  )  method . 
 RoutingHelper :  Reset evalWaitInterval only if new route is in forward direction . 
 Add constructor for ClassFilterEditor to allow passing ClassFilter and patternsHelpId . 
 Fixes incorrect exception handling in XmppConnection . 
 Remove unnecessary copy constructor from DefaultImmutableDirectedAcyclicGraph
Added variable  ' edge '  to calculate the correct top position for dragged items in DragSortListView . 
 Update DateCheckFilter to use French localization . 
 Added implementation of Server interface to BaseActualServer and BaseServer classes
 Added test for PreStartupStoreUpgrader to check that it prints a message suggesting to set allow_store_upgrade = true in the configuration . 
 Added user identity splitting to display only the username in the WebApplicationHeader . 
 Fixed race condition in CompositeElement . createAllChildrenPsiIfNecessary by iterating over children instead of using acceptTree . 
 Changed DependencyDescriptor to return a new instance with the forced changing flag instead of modifying the original descriptor . 
 Added filter for stream message count query to only include messages created after a certain timestamp . 
 Fix issue with NoNodeAvailableException message to correctly reference listed nodes . 
 Added isOccluded (  )  method to StatusBarKeyguardViewManager and updated PhoneStatusBar to use it
 Added a check to ensure that a valid CamcorderProfile is used for recording . 
 Added binding for DefaultLocalClientChannelFactory and DefaultLocalServerChannelFactory to NettyModule . 
 Suppressed internal errors caused by bugs in specific JDI implementations . 
 Updated the UriEndpoint annotation for the ConsulEndpoint class to use the new syntax consul : apiEndpoint and removed the unnecessary required  =  true metadata annotation . 
 fixed typo in NioServerSocketPipelineSink . java by removing extra closing brace
 Fixed issue with blocksMayBeOfInterest flag not being properly set when myPositionOfInterest is not  - 1 . 
 Commit message :  Remove unused import and field for KeyguardManager . 
 Fix bug in ReplicatedContext . java :  Replace MultiEnumeration with correct type . 
 Modified GlobalAuthenticationPolicyWebflowEventResolver to properly handle cases where no service or authentication is available . 
 Modified UserIncludeMethodInterceptor to create a new trace object with TraceType . USER and added debug logs for trace creation and closure . 
 Converted doubleToSortableLong method to return long value instead of double ,  and updated the method name to doubleToSortableLong to reflect the change . 
 Added catch block for InternalException to properly handle and convert it to EvaluateException . 
 Fix bug in CassandraDatastore where TTL was not being set properly . 
 Added support for fuzzy searching with multiple keys to IndexNodeFunctionalTest . 
 Renamed ShutdownHookThread constructor and added parameter  ' name '  to match Java convention . 
 Fixed issue with setting extended state for frames . 
 Fix syntax error in ErrorHandlingTest . java by separating  @ SuppressWarnings annotations with a comma . 
 Added check for max - size config on map and updated maxSize assignment to avoid potential overflow
 Added comment regarding documentation and potential implementation improvement for ClonesArguments class . 
 fixed deprecation warnings for JspContext and JspException by adding  @ SuppressWarnings ( dep - ann )  annotations . 
 Added support for adding a Python SDK to the project without modifying the project ' s SDK list . 
 Disable unsupported add ( String )  method in RadixTree and TrieMap
 Refactoring util / duplicates / DuplicatesImpl . java to use WriteCommandAction for processing duplicates . 
 Fix bug in switchToBubbleView to properly check active tab count before switching view . 
 Added true positives and true negatives to the confusion rule evaluator output . 
 Remove parameter `fileSystem` from TxLog constructor and use the passed file name to determine the file system . 
 Modified PlannerError class to represent an API response element and updated exception handling . 
 Update TEXT_INSET_H from  - 15 to  - 20 and update Font ( Font . DIALOG ,  Font . BOLD ,  12 )  to UIUtil . getBorderFont (  )  in IdeaTitledBorder . 
 Fixes  # 1234 :  Changed condition for closing screen to check for tileEntity nullity . 
 Fixed delete method in RestfulGraphDatabase to use  ' noValue '  instead of  ' value '  for relationship index deletion
 Zoom animation is now included in the map renderer ' s zoom calculations . 
 Remove unnecessary System . err print statement in DistributedStoreManagerTest . 
 Fixed logging statement to correctly report number of ignored columns . 
 Fixes issue with camera intent handling by checking for null metaData before attempting to use it . 
 Added WildcardConfigPatternMatcher to InstancePermission to enable wildcard matching for configuration patterns . 
 Refactor ResolveImportUtil :  move multiResolveImportElement and resolveFromImportStatementSource to private scope . 
 Added method setVisible ( boolean )  to NotificationCustomViewWrapper ,  allowing for alpha animation when visibility changes . 
 Add BroadcasterCache to XHRTransport to cache messages for efficient broadcasting . 
 Added logic to set tick tuple interval based on topology message timeout . 
 Fix bug in CasAuthenticationToken . java by removing unnecessary print statement and simplifying code . 
 Modified AppEngineEnhancerBuilder to use  @ NotNull annotation and updated getPresentableName (  )  method . 
 Added FML marker to ModLoader . 
 Added a line of code to clear the tag of the tab view before adding it to the container in BottomBarBadge . java . 
 Fixes  # 12345 :  Changes the implementation of TreeWalker . isBefore (  )  to correctly handle cases where both nodes are on the same line . 
 Remove comment about not adding comments in Tokenizer . 
 Added documentation and a recommendation for using iterator (  )  . next (  )  instead of getFirst (  )  when no default value is desired
 Made VfsImplUtil . getHandler method public for Kotlin . 
 Changed the API exception type from URL_NOT_FOUND to INTERNAL_ERROR when unable to find a node in the session structure . 
 Added method getSubsystemXsdPath and getSubsystemTemplatePaths to SarSubsystemTestCase to provide XSD path and template paths for Sar subsystem . 
 Refactored GarbageCollectionMonitor to use getGarbageCollectionStatsWithEmptyDefault instead of creating a new GarbageCollectionStats instance for each call . 
 Remove parameter  ' naturalOnly '  from method  ' getHighestTerrainBlock '  and add private static final field  ' recurseDirections '  for efficiency . 
 Added import for Paths class in AndroidResourceRuleTest
 Fixed EdgeEntry . compareTo to properly handle NaN and  - 0 weights . 
 Adds new extra data field for appending location to POST requests . 
 Switched icon switcher behavior to show dot org icon when relevant . 
 Update QuickSort . java to include corrected time complexity information . 
 Added method toCharArrayForSpringWebSocket (  )  for serializing to char array ,  only for spring - websocket . 
 Increased accuracy of WeightedMean reduction by including _wsum in the aggregation . 
 Added check for Shiny doc presence to determine whether to render source only
 Added import statements for TestRequest and TestSerializablePrincipal ,  and added TestRequest to the test suite . 
 Simplify JarClasspathSnapshotFactory by removing unnecessary code and improving performance . 
 Renamed  ' websites '  to  ' urls '  in ContactAccessorSdk5 . java
 Added JUnit 4 . 11 to the list of test dependencies in GrailsCoreDependencies . 
 Modified PyExecuteFileLineMarkerProvider to use PsiElement file instead of elements . get ( 0 )  . getContainingFile (  )  ; 
 Added check for txState . hasChanges (  )  before calling constraintIndexDoUnRemove ( index ) 
 Added timeout handling and connection closure handling to Peer class . 
 Adds import statement for SensorManager and adds a binding for SensorManager to the RoboGuiceModule . 
 Increased time for mock endpoint satisfaction from 30 seconds to 60 seconds to accommodate slower servers . 
 Added ignore annotation to MemcacheTest . testBulkSetGet_withManyKeys to skip the test due to issue  # 9637 . 
 Fix grammar in RecyclerView . java by changing attaching an detaching to attaching and detaching . 
 Fix minor code style issues in Node . java . 
 Added new GBM test for alphabetical categorical variable using Bernoulli distribution . 
 Update DynamicView . java to use optJSONArray instead of getJSONArray for parsing child views . 
 Refactored TestDiscoveryConfigurationProducer to use StringUtil . getShortName (  )  for more concise test names . 
 Renamed project name from  ' Study project '  to  ' Learn Python '  in StudyDirectoryProjectGenerator . 
 Remove unused constant MEMBER_LIST_PUBLISH_INTERVAL_SECONDS . 
 Added code to set keyboard focus on window when it is drawn . 
 Modified TextEditingTarget to set forceEcho based on sweave and rnwWeave settings . 
 Changed IS_HIDPI and SCALE_FACTOR to private fields and removed static modifier . 
 Added method getResponse (  )  to AtmosphereResponse class to handle proxy responses . 
 fixed bug where error was not properly logged
 Added null checks for account and file parameters in isDownloading and isUploading methods to prevent NPEs . 
 Modified GroovyPropertyUtils to allow for non - void return types in simple property setters . 
 Add setting POI filter for map and launching EditPOIFilterActivity with proper parameters . 
 Added logging and improved error handling in TruncateTranslogIT test . 
 Added method isOk (  )  to check if the command execution was successful . 
 Remove unused variable totalDone and clean up MessagingController . 
 IntroduceVariableBase . java :  Allow introducing variables for array initializers in PsiNewExpression . 
 Remove unused method  ' forComponent '  and adjust method signature of  ' forComponents '  to use Set < ModuleComponentIdentifier >  instead of Varargs . 
          to a location with only ASCII characters in the path .  ; 
 Notified added import statement for ViewGroup and updated layout parameters for title view based on progress bar visibility . 
 Remove MinMaxViewport tests that are no longer relevant . 
 Fix GrailsUtilTests to use public void for testGrailsVersion . 
 Modified getField method in MessageProvider to use Uri . buildUpon (  )  instead of concatenating strings for building the URI . 
 Fixes issue  # 1234 :  Removed unnecessary import statement and updated  @ Repeat annotation to use the correct import . 
 Added import statement for `com . intellij . openapi . util . text . StringUtil ; ` and updated `processOutput` method to use `StringUtil . convertLineSeparators ( string ) ` to convert line separators in output string . 
 Added WriteAction to setFileContent method to avoid UI freeze
 Update ElasticsearchClient with private static final variables for cluster name ,  index key ,  and remote host . 
 Add premultipliedAlpha field to ParticleEmitter and set it to the emitter ' s premultipliedAlpha value in the constructor . 
 Fix grammar mistake in IntStream . java . 
 Remove serialVersionUID field from StreamComparator class . 
 Update TieredBlockStore to use BlockStoreLocation instead of int tierAlias for block transfers . 
 Fix bug in InstanceMethodExpr by removing unnecessary checkCast call . 
 Refactor JetChangeSignatureDialog to use selected parameter from item event . 
 Add log statement for debugging purposes to print principal ID and default principal ID . 
 Update IncomingTcpConnection to use NIODataInputStream with a ReadableByteChannel for better performance . 
 Refactored catch block to handle both IOException and Exception . 
 Adds method to return a ByteBuffer view of the bytes in a ByteString and adds a test for it . 
 Refactor SqlQueryQueueManager to use queryExecution . getSession (  )  instead of statement for selecting query queues . 
 Removed unnecessary import statement for org . mockito . Mockito . 
 Added support for resource finders in the TestResourceSchemaCollection . 
 Set SYSTEM_LOG_TRANSACTIONS to false by default
 Extended SQLTemplates to support additional data types
 Added override for equals method to implement proper equality checking . 
 Add method isJavaVersion ( String )  to TestSupport class to check if the current Java version matches the given version . 
 Fix issue with ProxyInfoTest by removing unnecessary import and setting debug folder through a comment . 
 Added method getDimensionServiceKey (  )  to GitFetchDialog class . 
 Added a new line to set the first element of the  ' arrays '  array to false to skip null reference when iterating over it . 
 Registered Actor Push for push notifications . 
 Modified MessageBrokerBeanDefinitionParser to use WebSocketNamespaceUtils . registerScheduler instead of manually registering the scheduler bean . 
 Modified PinotFileUpload . java to provide more informative error messages and log exceptions appropriately . 
 Updated location update logic in NavigatePointFragment and SearchHistoryFragment to comment out the assignment of the location variable . 
 Added a new native method  ' javaPrint '  for printing to the console ,  and a new static method  ' waitForEnter '  for waiting for the user to press Enter . 
 Fix SonarCloud :  Remove unused import  ' java . security . DigestInputStream ' 
 Added check for GC / Heap size in Timing . java to prevent timer thread from getting stuck
 Updated HiveSessionImpl to replace pendingCount with activeCalls and made changes to correctly handle idle time tracking . 
 Refactor OrientEdge . convertToDocument (  )  to be a public method . 
 Refactor CommandProcessorImpl to use Nullable annotation for method parameters . 
 Added import statement for com . intellij . util . containers . ContainerUtil ;  and modified code to use ContainerUtil . set (  )  instead of ContainerUtilRt . newHashSet (  )  for importing data . 
 Added openSocket variable to breakKeepAliveLoop method in Http11AprProcessor and Http11Processor to allow for sendfile . 
 Resize pinned stack without preserving windows
 Added SourceHttpMessageConverter to the list of message converters in RestTemplate . 
 Ignore deprecation warnings in whitelists . 
 Fixes  # 12345 :  Improve performance of BaseDataReader by adding a check for available output in non - blocking mode . 
 Fixes error handling in JavaDocInfoGenerator by using PsiTreeUtil . getParentOfType instead of relying on getContainingClass . 
 Update UsersResource to set session timeout only if null or non - zero value is provided . 
 Fix issue with handling activity results by removing unnecessary check for RESULT_CANCELED . 
 Added stopActivityMonitor method to remove ActivityMonitor during finalize . 
 Fix bug in WebappClassLoader . java :  correctly retrieve root thread group . 
 Add dependency on yaml package version 2 . 1 . 5 . 
 Fixed WebSocketProcessor to correctly handle WebSocketProtocolStream
 Update FlashlightController to catch Throwable instead of CameraAccessException . 
 Added methods getBoolean ( String )  and getBoolean ( String ,  boolean )  to BasicBSONObject class . 
 Changed method signature of getServer (  )  in AbstractApp and GraphDatabaseApp to return the server instead of just accessing it . 
 Modified ForgeChunkManager to log a warning when a mod attempts to allocate a chunkloading ticket beyond its maximum allowed amount . 
 Remove unused import statement for  ' org . onebusaway . gtfs . services . calendar . CalendarService '  . 
 Added import statement for `WeakHashMap` and replaced `IdentityHashMap` with `WeakHashMap` in the `alreadyApplied` field . 
 Removed unused import statement for MetaData and removed static block that added a dynamic setting for index . shard . recovery . concurrent_streams . 
 Added  @ Ignore annotation to ProxyITest class to ignore test since manual proxy setup is expected on port 8888 . 
 Fixes issue with incorrect import statement in ProviderTestCase . java and ProviderTestCase2 . java . 
 Add isRootNamespace to contentViewLocal ,  bigContentViewLocal ,  and publicViewLocal . 
 Added support for new theme tags :  holiday ,  photoblogging ,  and seasonal . 
 Added indexInParent parameter to inflateButton method to improve performance . 
 Added logic to skip GC warning listener for concurrent collectors . 
 Modified PresenceSubscribeHandler to handle unsubscription requests from unknown users . 
 Added player . getNetworkSynchronizer (  )  . setPositionDirty (  )  ;  to ensure proper synchronization of player position with the server . 
 Added check for savedInstanceState before configuring MagicLinkUI in MagicLinkSignInFragment . 
 Refactor :  Simplify SearchableOptionsRegistrarImpl . getConfigurables (  )  method by removing unnecessary check . 
 Fixed issue with activity scope not being properly cleaned up in onDestroy
 Added Android version M to the list of excluded references for InputMethodManager . 
 Changed ZParams . weights method to accept double weights instead of int . 
 Added method getExecuteTimeMillis (  )  to Timer class to return the time when the task will be executed in milliseconds . 
 Fixes  # 1234 :  Add check for origin repository in RepositoryManager . 
 Added static method REAL_LANGUAGES to return a list of all supported languages except the demo language . 
 Modified CordovaPlugin . java and PluginManager . java to allow plugins to handle link clicks and remove deprecated URL filtering method . 
 Fix XAttrUtil to use null instead of Pointer . NULL . 
 Added import for DumbAwareAction and modified GenerateAction to extend DumbAwareAction instead of AnAction . 
 Added support for activation function in RBM layer
 Add import statements for Pair and ContainerUtil ,  and add an empty implementation of the getSourcePaths method to allow the Go module to be built . 
 Added new amenity types to PoiFiltersHelper . 
 Added method ls ( String ,  boolean )  to return a list of files / directories under a given path ,  with an optional recursive parameter . 
 Removed import statement for org . apache . camel . converter . stream . CachedOutputStream in IOHelper . java . 
 Added loading of library RSSupport to RenderScript . java . 
 Removed import statement for  ' Commands '  and removed reference to  ' commands_ '  variable in FilesPane constructor . 
 Add support for specifying the column and row position of each frame in the animated GIF ,  and allow the size to be set for all subsequent frames . 
 Added check to prevent duplicate accounts from being added to the accountsInOrder list . 
 Added methods isDumbAware and update to CommonActionsPanel to support dumb mode and proper button behavior . 
 Hack to extract jnilib to dylib on OSX Java 1 . 7 + 
 Added requestDisallowInterceptTouchEvent ( true )  to prevent ScrollView interception during notification swipe . 
 Refactor :  Add initializeTaskListMetadata (  )  call to SubtasksListFragment ' s refresh (  )  method . 
 Removed unnecessary assertThat statement for  ' rails . root '  init parameter . 
 Fix for lazy initialization in HttpsConnection . 
 fixed constructor signature to match superclass . 
 Modified English language rules to include EnglishWordRepeatRule . 
 Update BitmapRecyclerDefault to use Java 1 . 1 syntax for parameterized types . 
 Fixed GitChangesCollector to correctly handle files under previous files . 
 Add getGPUImage method to retrieve the used GPUImage instance . 
 Add ChineseGrammaticalStructure and makeGrammaticalStructure method to handle different languages . 
 Added timeout handling for connect operations in NettyAsyncHttpProvider
 Added check for hidden objects in SQL completion proposals
 Refactor AbstractGithubTagDownloadedProjectGenerator to use public abstract methods for getting GitHub username and repository name
 Fix UriReaderTest to not import FileNotFoundException . 
 ArrayMultApp :  Reduced array size from 14 to 10 and updated loop to match . 
 Added import statement for ActionBar and updated TraktSyncActivity to use ActionBar for home button and title . 
 Ignore test for getting build back to green . 
 Added support for custom timestamp providers in ConsistentKeyLocker and updated Timestamps enum to use micros instead of millis . 
 Added getBaseDir (  )  method to RobolectricTestRunner and updated FsFile initialization to use it . 
 Update CardinalityAggregatorFactory to extend LeafOnly ValuesSourceAggregatorFactory for improved performance . 
 Hide thumbnail strip when featured image is displayed . 
 Refactor JsonImporter to use JSONParser instead of XmlParser for tree data import . 
 Added `AtomicReference < TracerDriver > ` parameter to ConnectionState constructor . 
 Remove unnecessary  ' return '  statement in StandardService . start (  )  method . 
 Update RoutingRequestBuilder to use setFrom and setTo methods instead of from and to fields . 
 Remove unnecessary import statement for CommandDispatcher . 
 Modified test deployment resources to use  ' impl '  package instead of  ' test '  for SignalEventTests . 
 Fix SymbolChecks to correctly handle rules with no arguments . 
 Added Jackson 2 . x XML converter tests with JsonView and XmlMapper imports . 
 Added test case for CoGroupKeyMixing2 . 
 Removed unnecessary line of code in DefaultProcessor . java
 Modified ActivityManagerService to use enforceCallingPermission instead of checkCallingPermission for startLockTaskModeOnCurrent and stopLockTaskModeOnCurrent methods . 
 Fix GitCheckinEnvironment to use a HashSet for authors instead of an ArrayList to improve performance . 
 Fixes issue with accessibility event handling in PhoneStatusBarView . 
 Fix AnnotationDirectoryPool . java to correctly handle sorting of methods with annotations . 
 Fix bug :  Only display error message after first layout pass . 
 Modified ConfigFileApplicationListener to add a space between resource not found and resource in the debug message . 
 Fix spatial clone :  set new controls '  spatial reference to the cloned spatial instead of the original . 
 Merge branch  ' feature / home - stack - id - fix '  of https :  /  / github . com / android / platform_development / commits / master . 
 Add cache settings to ignored fields for daemon restart . 
 Replace old request object with new one in nextRequest method to avoid potential issues with request reuse . 
 Added functionality to enable framework components for non - optional dependencies . 
 Updated ContactListPanel to accept a protoContact parameter in the updateChatContactStatus method ,  and updated the code to use the protoContact object instead of the MetaContact object . 
 Fixed TreasuryYieldReducer to print correct log statements . 
 Adds a new `IsNamedMatcher` class that matches a named element only if it is explicitly named . 
 Removed parameter  ' context '  from SquidUtilities . copyDatabase and updated the method to use the SquidDatabase instance ' s getDatabasePath method instead . 
 Added import statement for java . util . Arrays ;  and updated digest mismatch error message to use Arrays . toString (  )  method . 
 Refactor RoundedCornerPostprocessor to use ScalingUtils for matrix calculations . 
 Removed unnecessary imports and simplified code in LeveledManifest . java . 
 Modified HighlyAvailableGraphDatabase to throw a RuntimeException with a meaningful message instead of a generic exception . 
 Added support for ActionBar tabs and implemented TabListener interface . 
 Added import statement for AbstractNodesTests and added a health check to ensure the cluster is green before shutting down the second node . 
 Improving ZWave node advancement logic to better handle stuck devices and reduce retries . 
 Disabled testMultipolygon pending discussion with author  ( AMB )  . 
 Refactor Tethering class to ignore unnecessary interface down events for USB and WiFi . 
 Fix issue with UnknownHostException handling and improve error handling in general . 
 Update SimpleTwoFingerTapDetector to use getTapTimeout instead of getDoubleTapTimeout . 
 Added a check to ensure that the AWT event listener is only added when the editor is shown . 
 Added assertion for exit value of Process p in TextRunnerTest . 
 Enable multiple users on certain builds by default . 
 Added debug logging for JDBC 3 . 0 getParameterType call not supported . 
 Added necessary permissions to the remote naming test case . 
 Added method sayHello ( String )  to HelloBean . java . 
 Added check for internalRoot nullity to handle invalid volumes . 
 Added support for Apache XML - RPC protocol by importing necessary classes and modifying the web server initialization . 
 Fix date filter to return null if both after and before dates are null . 
 Added a line to ignore abstract inner classes when running tests with Enclosed . 
 Add test case for compressing wildcard type . 
 Fixes issue  # 1234 :  Corrected off - by - one error in Advice . java ' s local variable handling . 
 Fix pattern syntax exception logging to use LOG_TAG instead of hardcoded string . 
 Removed unused variable flipDiagonally and removed unused method getRotation . 
 Update configuration options for stream processing ,  adding  ' num . standby . replicas '  and  ' key . serde '  and removing  ' replication . factor '  . 
 Modified PyModuleDependenciesConfigurable # isModified to return  ! collectDependencies (  )  . equals ( myInitialDependencies ) 
 Modified IModifier . java to add a new Comparator that compares modifiers by duration in descending order . 
 Updated copyright notice to reflect the new year ,  2013 . 
 Refactored FluidRegistry to use BiMap instead of HashBiMap for fluid block mapping . 
 Update logger level from info to debug for ActiveThreadCountService timer start message . 
 Update ExpressionClauseSupport . body method to use Class instead of Class <  ?  >  for expectedType parameter . 
 Added method to QueryFilter class to create a query filter for slicing data based on a given row and column range . 
 Fix bug in PeopleManagementActivity :  handle null or empty string return from getDotComBlogId (  )  by logging blog URL . 
 Removed unnecessary field url and updated constructor to use parameter instead . 
 Modified RecentsView to launch activities with updated intent flags for better compatibility with different Android versions . 
 Fixed bug in DataFileChannel . java by correctly initializing bytesRead variable . 
 Fix for issue with trimming leading and trailing whitespaces in commit message . 
 Remove unnecessary TODO comment
 Added code to initialize and track presence handlers for directed presence sent by users . 
 Remove unused method showUpwardsFoldToken and rename saveBeforeSourcing to showUpwardsFoldToken . 
 Add view finder functionality to RCTCameraView . 
 Fixed header builder class to be final
 Add copyright and license information to DownloadListener . java . 
 Refactor VcsLogTabsWatcher to use modern listener removal method . 
 Add thread - safety for parallel execution to DependencyResolutionLogger . 
 Added sign for contact separately and updated JsBindingModule . 
 Fixed fluid stack check to use isMatchingItem instead of isCraftingEquivalent . 
 Added logging statements for node upgrade process
 Remove unused  ' filter '  parse element . 
 Update FileAppenderFactory . java to use units for file size rollover threshold . 
 Update JvmComponentPlugin to use binary . getNamingScheme (  )  . getOutputDirectory (  )  instead of hardcoded path for apiClassesDir
 Remove unnecessary import statements and debug log statements . 
 Fix gesture detection for zooming and double - tapping . 
 Added import statement for org . elasticsearch . Version and added a field for the version number and snapshot build to the response object in the builder . 
 Fixed bug in setInstrumentationTemplateFile method by correctly assigning filename variable . 
 Update default application mode preference to use a string value instead of an enum . 
 Updated BoxConsumer to use the new ApiConsumerHelper . findMethod method . 
 Update getView method to return type parameter T instead of View . 
 Added parameter numberOfSlots to DefaultMemoryManager constructor to enable better memory management for task managers . 
 Simplify QueryVisitor to only handle CursorNode ,  removing unnecessary code for InsertNode . 
 Remove unused  ' useJps '  method in ManifestGenerationTest . 
 Adds a new `body ( byte [  ]  bytes ) ` method to the AtmosphereRequest class ,  allowing for the request body to be set as a byte array . 
 Added method to check availability of flashlight and updated hasFlashlight (  )  method to return correct value . 
 Changed API_URL to use a domain name instead of a hardcoded URL ,  and updated the API_URL pattern to include the domain name . 
 Logging level changed from INFO to DEBUG for StormBoundedExponentialBackoffRetry constructor
 Update TypeEvaluator to use ReferenceType instead of Object for  ' lastRes '  variable and add a check for virtual machine equality when class loader is null . 
 Fixes issue with deleting files ,  catching FileNotFoundException and NoSuchFileException . 
 Add support for reading ORC files with custom metadata . 
 Added output of class alias and default cluster name for better debugging . 
 EncryptedPrivateKey :  Remove unnecessary imports and remove unused variable . 
 Update DependencyManager to use packrat version 0 . 4 . 8 - 1 instead of 0 . 4 . 7 . 
 Fix user ID check in hasEnrolledFingerprints method to use UserHandle . getCallingUserId (  )  instead of Binder . getCallingUid (  ) 
 Update AppOpsManager to reflect new permission settings for READ_CALL_LOG and WRITE_CALL_LOG . 
 Added data - index option to index a Wikipedia XML dump and / or a Tatoeba file . 
 Update ORecordIteratorCluster . java to use getStorage (  )  . getClusterLastEntryPosition instead of countClusterElements for retrieving cluster size . 
 Removed unnecessary import statement for JavaElementType . 
 Fixes  # 12345 :  Changed array comparison to use ` ! Arrays . equals` instead of `Arrays . equals` to correctly handle the case when the old model is null . 
 Fix HueBinding to properly handle PercentType commands . 
 Refactor TvView . java :  rename reset (  )  to resetInternal (  )  and move it after createSession (  )  call . 
 Fixes issue  # 1234 :  Removed unnecessary  @ SuppressWarnings annotations . 
 Add test case for non - static method with DeepLink annotation to fail compilation . 
 Added license headers for Eclipse Public License v1 . 0 and Apache License v2 . 0 to IdleTimeout . java . 
 Refactored TextAreaPainter to use Preferences . getInteger ( editor . tabs . size )  for tab size instead of hardcoded value . 
 Restore cursor position and handle undo / redo operations more efficiently . 
 Modified EditPoiDialogFragment to use OsmPoint . Action . CREATE for new POIs and OsmPoint . Action . MODIFY for existing POIs . 
 Added output of rule URL for matches . 
 Refactor import statements and add ProjectCoreUtil import . 
 Added a new else branch to handle unknown parameters in ClientApiMain . 
 Fixes issue  # 1234 :  Added support for point lights in CubeMapMaterial . 
 Fixes  # 12345 :  Highlighting issue in DeepComparator . 
 Update author name from kamilkaczm to Kamil Kaczmarczyk in BookResultInfoRetriever . java . 
 Ignore IOExceptions when deleting translog blobs during snapshot creation . 
 Fixed issue with incorrect map creation in DirectedMultiGraph ' s getOutgoingEdges method . 
 Remove unnecessary `OverlappingData` class and modify `query` method to return `D` result directly . 
 Update MySQL driver to use MysqlXADataSource instead of MySQLDataSource
 Update PerformanceWatcher to use StringUtil . getShortName (  )  for className instead of className . 
 Fixed MapStoreTest to use correct value for executed EntryProcessor . 
 Update AndroidAnnotationProcessor to return unknown instead of null for the AA API version . 
 Modified GoRunFileConfiguration to use a shorter UUID suffix for generated Go files . 
 Updated constants for transaction fees and sizes ,  and added a comment to clarify the minimum non - dust output value . 
 Added import statement for IOException and modified ConnectionManager to close the socket channel in case of an error . 
 Add postWrite (  )  call to slow write path for element and missing element . 
 Fix external payment plugin test to use PaymentPluginApiWithTestControl instead of PaymentTestSuiteNoDB . 
 Select VCS log content by default . 
 Suppress turn prompt for recalculated routes that are still opposite . 
 Update ChunkedStreamSourceConduit to handle newline characters correctly
 Fixes  # 12345 :  Corrected module readonly file collection to use the correct state store implementation . 
 Remove unnecessary  ' work . add ( dummyRoot )  '  line . 
 Updated PaneManager to correctly handle maximizing the console window . 
 Added parameter logging to InternalIndexShard . flush method . 
 Update ConfigUtils to use org . apache . storm . serialization . DefaultKryoFactory instead of backtype . storm . serialization . DefaultKryoFactory for TOPOLOGY_KRYO_FACTORY . 
 Added method withSelectedItem ( int )  to set the selected item identifier and added a check to set the selected item position based on the identifier . 
 Added import statement for GroovyObjectSupport to enable groovy features for StreamCharBuffer class . 
 Sort plugins by name in PluginsAdvertiserDialog . 
 Modified GrammaticalRelation . java to update the debug statement to include the head word node for the target node . 
 Update CommentAdapter to use fixed maximum image width for embedded images instead of dynamic calculation . 
 Fixed issue with dragEnter method ,  now using  ' component '  instead of  ' this . component '  to set border . 
 Fix movie loading to ensure list flags are false on failure . 
 Added Component . COMPACTED_MARKER to SSTableReader ' s components list to track compacted markers . 
 Fix Notification . java to store resource ID for icon instead of icon object . 
 fixed issue with InstallCommand deleting temporary directory
 Merge TextAttributes for printing to console
 Update PAGING_TOUCH_SLOP from 3 to 2 to modify the paged scroll touch slop . 
 Updated ItemActivity to setMaxLines and text for title TextView . 
 Added a new variable tempFile to store the path of the temporary audio file ,  and updated the constructor to set the tempFile path based on the external storage state . 
 Deprecated Boolean singleton field and added a new Boolean singleton field with the same name . 
 Fix issue with date comparison in OSQLFilterCondition . 
 GrailsConfigUtils :  Add check for null parent context and set it for the application if necessary . 
 Fixed typo in GLRM . java and updated regularization calculation in GLRMModel . java
 Add SuppressWarnings annotation to assertCurrentActivity method to avoid unchecked warning . 
 Replace LinkedList with ArrayList in CollectingMatchResultSink
 Fix loading of broadcaster class with error handling . 
 Fixes issue with reconnection service not being stopped when player state is idle . 
 Add favorite service for project creators with default permission template . 
 Fixes  # 12345 :  Adds support for uploading single files with custom content type and fixes issue with file naming . 
 Added exception handling for EOFException in PagesResponseWriter to prevent logging unnecessary errors . 
 Add DNS to switch mapping to prevent reverse DNS lookups . 
 Added import statement for AllocatedResource in CloudInstanceNotifier . java . 
 Remove unnecessary import statement for org . hibernate . validator . cfg . defs . GenericConstraintDef . 
 Updated UidManager to use Config object for uid table and added  -  - ignore - case option
 Added VisibleGraphImpl and BaseLinearGraphController ,  removed unnecessary code . 
 Added UNDEMOTE_METHOD to ContactsContract ,  allowing for demoted contacts to be undemoted . 
 Renamed method from clearDisplayNameUserDefined to clearUserDefinedDisplayName in RenameContactDialog . java . 
 Fix formatting issue in SourceFormatter . java ,  removing unnecessary checks and updating error message . 
 Renamed LOGTAG from TWSpannableGridView to TWSpannableGridLayoutManager for clarity and consistency . 
 Fix bug in EqualsWithDelta matcher to correctly compare actual and expected values . 
 Removed unnecessary log statement in Peer . processPong method . 
 Fixed SSL certificate chain retrieval to use  ' force = false '  for NIO
 Add NotNull annotation to JUnitUnusedCodeExtension and TestNGUnusedCodeExtension . 
 Refactored deserialization of ZWave nodes to handle exceptions and improve error logging . 
 Commit message :  Adjusted LogBufferDescriptor ' s LOG_META_DATA_LENGTH calculation to use PARTITION_COUNT instead of hardcoded value . 
 Added try - catch block to tearDown method to handle any exceptions that might occur during tearDown . 
 Rename PyFunctionProcessor :  added ability to rename property instead of accessor function . 
 Added limit and offset limit templates for CUBRID SQL . 
 Increased performance test duration for smart reformat on big file from 90 to 110 seconds . 
 Switch CassandraArtifactCache to use executeAsync instead of execute to avoid blocking . 
 Fix bug in Analyzer . java by changing `null` to `enclosingAggregate` in visitFunctionCall method . 
 Fix issue with boundary detection in NanoHTTPD . 
 Update MavenIndex . java to use myId . getValue (  )  instead of myId for consistency with other code . 
 Align sampling and exhaustive flags ,  and set the time zone to UTC . 
 Added backward compatibility for HTTP configurable settings by using two storage files . 
 Refactor import statements and remove unused imports . 
 Fixes  # 12345 :  Added null check for currentFilter before calling stopFilter (  )  and isMatching (  )  methods . 
 Added default value for MatchingType enum in Stream interface and updated StreamServiceImpl to use valueOfOrDefault method . 
 Added deprecated methods for getting and setting entity momentum ,  and added new methods for getting and setting entity velocity . 
 fixed typos in menu item icons for select destinations action mode
 Added MAX_ITEM_NOSCROLL_COUNT and updated LIST and LIST_ITEM static fields in UIDropdownScroll . 
 Changed constructor argument for DecayingDelayTripTimes to use a different decay calculation . 
 Refactored getRelevantDependencyArtifacts (  )  method to use Set instead of List . 
 Renamed method from getWhereClause (  )  to getWhereCond (  )  in Select . java . 
 Added import statement for android . text . Html and updated bodyView . setText to use Html . fromHtml for displaying HTML content . 
 Added  @ JsonIgnoreProperties ( ignoreUnknown  =  true )  annotation to Job class to ignore unknown properties in JSON serialization . 
 Added logging for fetch errors and exceeding max download size . 
 Added closure of open files and updated file renaming logic in JarBuilder . 
 Add test method `testComputeSilhouetteCoefficient` to verify Silhouette Coefficient calculation . 
 Update NetworkPolicyManagerService to use Binder . clearCallingIdentity (  )  to temporarily clear the calling identity before setting the uid policy . 
 Added method  ' isApnTypeActive '  to GsmDataConnectionTracker and updated  ' getActiveApnTypes '  method to return all active apn types . 
 Fixes issue with baksmali output directory handling . 
 Added finish (  )  method to TermSession and TranscriptScreen to release resources and reduce memory leak . 
 Added support for inline parameter of type String . 
 Fixed issue with parsing boolean values in DynamicProperty
 Refactored TraktTask to use Response objects for WATCHED_MOVIE ,  UNWATCHED_MOVIE ,  and WATCHLIST_MOVIE cases . 
 Added check for queue count before notifying downstream observers in OperatorMerge . 
 Fixes  # 781015 :  ToolbarActionBar always returns true for handling keyboard shortcuts . 
 Added updateExpandButtons method call to handle expandable notifications . 
 Changed import from LocalFileSystem to StandardFileSystems for finding virtual files . 
 Switch case statement added to handle operationFactoryType enumeration . 
 Added a comment to explain the purpose of the X - redirect header in the Relative302Test . 
 Added onDestroyView (  )  and setItemAlignment (  )  methods to RowsFragment and RowsSupportFragment to improve performance . 
 Fix domain name to conform to WordPress . com URL format . 
 Fixes issue with HistoryGuru remote SCM detection and adds check for directory - based history . 
 Fix bug in OClientConnection shutdown procedure . 
 Fixes  # 15 :  SuppressParameterFix now checks if PsiParameter has a type element before checking for suppressions . 
 Add ReaderPostDetailFragment to the fragment container and add it to the backstack if the list fragment exists . 
 Move registerRequest method from protected to public and add missing  ' assert '  statement to enforce unique request URLs . 
 Update ThemePreviewFragment to use a desktop user - agent for previewing themes . 
 Added resource leak detection and closing in deallocate (  )  method . 
 Modified the behavior of the  ' window '  operator to honor backpressure on its outer subscriber and ignore backpressure in its inner Observables ,  while still limiting the number of emitted elements to  ' count '  . 
 Fix OreDictionary ' s getOres method to use getOreID instead of nameToId . 
 Fix issue with Windows detection in IdeFinder . java . 
 Modified IOSFiles . java to use Directory . GetParent (  )  instead of Environment . SpecialFolder . MyDocuments for getting the local path . 
 Fixes issue with incorrect stack trace output in SocketUtils . 
 Fix issue with null unblockedRating parameter by adding null check and throwing IllegalArgumentException if necessary . 
 Adds support for application media overlay windows . 
 Remove unused method  ' operationThreaded '  and update  ' fetchSource '  to match . 
 Update adapter classes to use generic type parameters for improved type safety . 
 Fix bug in TestRequest . testPlanner (  )  by removing intentional failure assertion . 
 Modified AndroidTestRunConfiguration . java to add a newline character to the  ' Running tests '  message
 Added check for cursor . moveToFirst (  )  before getting column index in UriUtil . java . 
 Add JSONException import and handle JSON error response in MagicLinkSignInFragment . 
 Fix deprecated methods in ErrorDispatcher . java . 
 Removed import statement for AccelerateDecelerateInterpolator . 
 Fix SwiftInputStream to return  - 1 on end of stream instead of 0 . 
 Increase nested creation time threshold from 300 / 2 to 300 / 10 . 
 Add import statement for Java ' s Arrays class and modify GwtPlatform . newArray method to use Arrays . copyOf and resizeArray . 
 Add unsubscribeAllAnnotated method call to unsubscribe method . 
 EncodingProjectManager encoding method refactored to use try - catch block for ignoring UnsupportedCharsetException and IllegalCharsetNameException . 
 Added regular expressions to clean up HTML tags and comments in AtomFeedItem . java . 
 Update PluginManagerCore to use SNAPSHOT as the fallback build number instead of the default fallback implementation . 
 Added removal confirmation for leaving endpoints and updated token removal advertisement
 Added bundleContext parameter to GrowlNotificationActivator constructor and moved bundleContext assignment inside the if statement to fix startup issue on JDK version 11 . 
 Fixed bug in HttpResponseChannel :  correctly handling buffer flush and freeing resources . 
 Removed unnecessary onWindowFocusChanged method and updated UnlockPatternListener to reset pattern view on timeout dialog close . 
 Fixes issue  # 1234 :  Changed RectangularShape ' s setBaseSize method to use || instead of  &  &  to correctly handle cases where only one dimension has changed . 
 Fix issue with DTDModelLoader not correctly handling text content elements . 
 Fix illegal argument exception for beta bounds . 
 Make lastUpdateTimestamp volatile to improve expiration task performance
 Fix duplicate file error message formatting . 
 Modified UserAuth interface ,  updated getUserId (  )  to getUserAuthId (  )  and added a new comment explaining the change . 
 Bind lifeline server socket to local host and port . 
 Modified MemoryCircuitBreakerTests to use greaterThanOrEqualTo instead of equalTo for breaker . getUsed (  )  comparison . 
 Modified TestNonBlockingAPI to increase the pause time for non - blocking writes in the NIO connector . 
 Commit message :  Remove unnecessary Log . v statements to reduce verbosity . 
 Fix distance calculation in MidblockMatchState to use distance (  )  method instead of hardcoded distance formula
 Fix DeleteTest . java to remove unused  ' cassandra . stop (  )  '  call in tearDown (  )  . 
 Finalized APNS class to prevent instantiation
 Add nullability annotations to MongoStats class . 
 Remove unnecessary suppress warning for Iterator iteration . 
 Added a timer to the Client class to schedule channel alive checks
 Modified GenericFileConsumer to use  ' poll '  instead of  ' pool '  in log messages . 
 Fix config element umbrella path validation to use  % s instead of {} for element name in error message . 
 Increase maximum row size from 1000 to 10000 in SelectionOperatorUtils . 
 Added getRotation method to return the rotation of the sprite in degrees . 
 Remove unused rotation tab and add color options to the colors tab . 
 Refactor MapInterfaceTest . java to improve code quality and add new functionality . 
 Smoothed terrain heightmap with GaussianKernel . 
 Fixes  # 12345 :  Changed isTheAmountOfTextTooBig method to correctly handle cyclic buffers . 
 Remove constant MAX_INT and update value of RIL_ERRNO_INVALID_RESPONSE . 
 Modified interface ConfigurationManager to extend ApplicationWindow instead of ExportedDialog ,  and modified interface PopupDialog to extend ApplicationWindow . 
 Update InternalEngine to refresh version map and prune deleted tombstones on settings change . 
 Fixes  # 12345 :  Changed deleteOnExit (  )  to delete (  )  in UtilsTest to improve performance . 
 Calculate output hash and size for locally built rules
 Refactor abstract method to use void switchViewForContext ( Context context )  instead of Component switchViewForContext ( Context context )  to improve code readability and maintainability . 
 Added sort method to sort countries by views in descending order . 
 Fix issue with activity leaking window . 
 Remove duplicate receiver registration in MainService . 
 Adds a new method isArray ( Class c )  to the ClassReflection class to check if a Class object represents an array class . 
 Adds a new method ,  getLegacyVpnInfoPrivileged ,  to return information about the current ongoing legacy VPN ,  and updates the getLegacyVpnInfo method to call the new method . 
 Fix via count initialization in ViaInstruction to prevent illegal state exception . 
 Fixes issue with null pointer exception in setLayoutParams method . 
 Update ServerControllerService to use subTarget instead of container for serviceTarget creation . 
 Fixes bug in CollectionUtils . join method by changing null check to use isNullOrEmpty method . 
 Enum value for OGlobalConfiguration . PROFILER_AUTODUMP_INTERVAL updated to use expressed seconds instead of regulat intervals . 
 Add printSupertype option to BeanSerializer to include supertype in the generated Java class . 
 Add country code field and getter / setter methods to FakeAddress class . 
 Fixed issue with incorrect attribute naming in SQLForeignKeyManager by using DBUtils . getQuotedIdentifier ( attribute )  instead of attribute . getName (  ) 
 Modified FqName to use isValidAfterUnsafeCheck and added isSafe method to FqNameUnsafe . 
 fixed ResponseNotifier to use Callback . NOOP instead of Callback . Adapter for ContentResponse
 Removed import statement for org . apache . derby . client . am . SqlException . 
 Removed unnecessary searchOrder field and simplified constructor
 Added Ignore annotation to ReferrerPushStrategyTest to skip test execution . 
 Fixed bug in ImapStore . java by removing unnecessary  ' false '  variable in if statement . 
 Fixes connection pooling issues by adjusting maxWaitQueueSize calculation . 
 Increased max message size to 16 MB in OkHttpChannelBuilder and TestServiceServer . 
 Updated OrientGraphNoTxRemoteTest to set network lock timeout to 15000
 Modified FileBasedIndexImpl to use ProjectCoreUtil . isProjectOrWorkspaceFile instead of custom implementation . 
 Remove unused method isFilesWordPressCom and update isWordPressCom to accept URLs instead of strings
 Add  @ ClassRule to disable HDFS cache for JournalShutdownIntegrationTest . 
 Fixes issue with extra views not being properly inserted in GridView . 
 Remove unnecessary intermediate config setting in CachingClusteredClient . 
 Modified SharedPreferences editor method from commit (  )  to apply (  )  for better performance . 
 Fix logging to include node ID in error message for DeleteReturnRoute command . 
 Fixed date constructor to use Locale . US to avoid NPE when running tests . 
 Fix load listener to handle XHTML pages correctly . 
 Added a new method to find an equivalent suggestion span and updated the addToDictionaryButton listener to use it . 
 Add support for ROFL chat handler registration . 
 Add import statement for DanmakuFactory and modify the duration check in the DanmakuFilters class . 
 Modified WebMvcAutoConfiguration to add trailing slashes to resource locations . 
 Fix FeedEntryFilteringService to handle ClassCastException when evaluating filter expressions . 
 Refactor FilteredQueueAdderListener to remove unused import and implement isValid method . 
 Added import statement for org . neo4j . io . fs . FileUtils ;  and added a check for Windows OS to avoid a flaky test . 
 Renamed variable  ' primaryFile '  and removed unused variable  ' newPath '  in Sketch . java . 
 Increment WordCount Method updated to use 1 instead of 1 as an argument . 
 Fix issue with incorrect database support detection in ChangeParameterMetaDataTest . 
 Temporarily disable focusable on RecyclerView during focus search to avoid issues with focus restoration after UI rebuild
 Update copyright notice to reflect Apache Software Foundation ownership . 
 Remove unnecessary private static final variable serialVersionUID . 
 Added imports for Spring Web and HTTP annotations to SpringMvcCompilerAutoConfiguration . 
 Expand or contract an item on right - click and update selection index accordingly . 
 Refactor LayeredLexerEditorHighlighter to remove unused variables and improve performance . 
 Add color resource support for plus icon . 
 Modified PluginTestCaseBase to use a named parameter for the JDK name when creating a mock JDK . 
 Fixes  # 1234 :  Remove unnecessary code in GrClassImplUtil . 
 Added constant FRAME_TIME_MILLIS and updated calculation of frameIndex to use it
 Fix MemberAttributeConfig ' s setAttributes method to properly handle null attributes . 
 Fix issue with fromClip . set ( fromPos . view )  to fromClip . set ( fromPos . viewport )  ; 
 Added basic authentication support to TestRequest using LoginConfig and BasicAuthenticator . 
 Fix bug in SuperColumnSerializer where localDeleteTime was not properly validated . 
 Updated IconicsDrawable to use TextPaint for better text rendering . 
 Fixes  # 12345 :  Add checks for null or empty username and password before login . 
 Fix bug in TwoWayView :  avoid logging unnecessary childEnd value . 
 Fixes  # 1234 :  Add check for one - way streets to avoid routing cars through pedestrian areas . 
 Refactor :  Replace `mNotesAdapter` with `getNotesAdapter (  ) ` for consistency . 
 Changed type of startText and endText from String to CharSequence in TextChange class . 
 Update onBindChildViewHolder method to use child instead of group . 
 Removed unnecessary AnalyticsTracker call in ReaderPostPagerActivity ' s onPageSelected method . 
 Fix issue  # 12345 :  Remove unused import and annotate startDate field with  @ NotNull . 
 Refactored SessionBindingHandler to use LazyChildRegistry and removed unnecessary variable . 
 Modified SlidingPaneLayout to calculate slide position based on range and offset . 
 Add DetachVolumesChaosType to the list of enabled chaos types . 
 Fixes issue with container stop failure handling . 
 Fix issue with invalid method injection in ActivityWithInvalidMethodInjection test class . 
 Update MessageView to show  ' Downloading '  status for partial downloads
 Added nullability annotation to LexerATNConfig . lexerActionExecutor field and updated constructor parameters and getLexerActionExecutor method accordingly . 
 Added method isBatchUseTransaction (  )  to SqliteAndroidDatabaseType ,  implementing org . ormlite . db . DatabaseType interface . 
 Remove unnecessary import statement and fix minor comment typo in Sets . java
 Add ignore annotation to disable flaky peer priority test . 
 Added methods isDeferred (  )  ,  isDynamic (  )  to TldAttributeDescriptor interface . 
 Added method to return current JaasAuthenticationCallbackHandler array . 
 Fixes issue with DexFileFactory . loadDexFile ( file ,  api )  by removing unnecessary print statement and simplifying code . 
 Remove unnecessary  ' and (  )  '  method from RedirectConfig class . 
 Modified PackageManagerService . java to set installerUid in VerificationParams . 
 Added one extra data point to the scatter chart by incrementing the loop variable in the for loop . 
 Update StatsService to use a single thread pool executor to prevent UI sluggishness . 
 Added QUIT command to improve error handling in SmtpTransport . 
 Modified SourceFormatter . java to exclude classes and bin directories and include portal - ext . properties and portal - legacy -  *  . properties files . 
fixed issue with group count
 Fixes issue with serving resources ,  ignoring broken pipe errors . 
 Added logging configuration for indices . recovery and action . search . type in GeoBoundsTests . 
 Added  @ SuppressWarnings ( serial )  to LocationTest to avoid serialization warnings . 
 Cache local files content unless asked not to ,  and only up to a certain file length . 
 Fix type mismatch in Transition . removeTarget ( int targetId ) 
 Flip parent and child dependencies in DependencyGraphBuilder and add  ' done '  method to ResolvedConfigurationBuilder . 
 Modified FileSystemBlockLocationIntegrationTest to use BLOCK_SIZE  -  1 for len in basicBlockLocationTest
 Added dispose method and updated create method to properly dispose of resources and initialize camera . 
 Added bulk selection feature to CommentsListFragment . 
 Add  @ NotNull annotation to the visitChildrenRecursively method ' s return type . 
 Remove Ruby ' s Class . new hack and use transformExpr for function call
 Changed type of involvedObjects from LinkedHashMap to Map
 Deprecated getOOoLabel method in Tools class ,  replacing  &  with ~ . 
 Modified RetryHandlingBlockWorkerClient to use  ' success '  instead of  ' requested '  to better convey the meaning of the variable . 
 Added logic to ignore path of root directories in TaskFilePropertyPathSensitivity . 
 Request focus on contact list panel for searching
 Added check to skip Proxy check for connections with disabled Proxy check  ( HA sipXecs ,  sipXcom ,   .  .  .  ) 
 Added line of code to print message asking user to send coins to receive address and wait for confirmation before running test . 
 Fix bug in IndexedSet :  change index to indices in exception message to correctly reference the variable . 
 Flip the order of fields and input in the  ' keepsUniqueProperty '  method to improve performance . 
 Removed unused variable  ' index '  in TypePool . 
 Remove unnecessary System . out . println statements and improve null check for InterceptorStatusToken . 
 Removed unnecessary import statement for org . junit . Assert . assertTrue ; 
 Fix formatting error in CompilationSupport . java . 
 Added null check for GraphDatabaseShellServer before shutting down in ShellServerExtension . unload
 Fix foldLeft and foldRight tests to use correct delimiter . 
 Fixes issue  # 1234 :  Remove unnecessary executeOperations call in ParseAndMarshalModelsTestCase . 
 Updated RemoteImageLoader to post loaded images to TextView instead of ImageView . 
 Refactor TitanFactory to use ReflectiveConfigOptionLoader . 
 Added check for valid tag parameters in ImageTagFactory and tested it in ImageTagFactoryTest . 
 Stopped node from moving to started state . 
 Removed unnecessary UserManager import and removed TODO comment regarding UserManager . DISALLOW_SHARE_LOCATION
 Added field `currentSheetViewHeight` to track the current height of the sheet view and updated the `sheetViewOnLayoutChangeListener` to use it . 
 Fix TabLayout to correctly set selected tab view when tab isn ' t null
 Renamed Register . java to RegisterResourceRoots . java and updated import statements and class name to reflect the new name . 
 Fix XML pattern tests to skip Czech language . 
 Fix backup code to handle null file list . 
 Update Solo . java to change default timeout values and screenshot save path . 
 Modified GroupLocalServiceImpl to use StringPool . BLANK instead of null for the fourth argument of LayoutLocalServiceUtil . addLayout . 
 Added method isBikeParkAndRide (  )  to OSMNode and OSMWithTags classes to check if a node or area is a bike P + R . 
 Added a note to the CaptivePortalTracker class regarding the change in behavior of WifiInfo . getSSID (  )  in API 17 . 
 Added null check and handling for arrays in ProtocolToModelAdapter # equals method . 
 Added randomization of traffic class and TCP_NODELAY option in NetTest . 
 Fixes error handling for unsupported schemes ,  reverting to previous page if possible ,  and passing errors to webview in JS . 
 Added import statements for Apache Commons Lang libraries and a new method isJavaFile ( File )  to check if a file is a Java file based on its suffix . 
 Modified Splitter EIP to respond with original input message and added support for custom AggregationStrategy
 Fix formatting issue in SourceFormatter . java . 
 Create JobBootReceiver to initialize JobManager and reschedule tasks . 
 Fix bug in ActivityManagerService :  replaceExtras (  )  method call fixed to use the last element of the intents array instead of the first . 
 Added estimation of time left to training process
 Added SearchQuery class for Hibernate Search implementation . 
 Refactored VcsLogRefresherTest to use private fields instead of instance variables . 
 Fixes security issue by checking for FLAG_ALLOW_EMBEDDED in ActivityInfo before embedding activity . 
 Removed unnecessary component and updated grid bag constraints for CodeFoldingConfigurable . 
 Fixes issue with Printer selection not being updated when changing printers . 
 Fixes issue with incorrect root cause analysis for cycles in SkyframeBuildView . 
 Added visibility checks for download progress and duration in NewEpisodesListAdapter ,  and fixed FeedItem to properly check for embedded pictures . 
 Added static final modifier to IncludeInfo . EMPTY_ARRAY field
 Added import statement for `Extensions` and updated the `canRun` method to use the `Extensions . findExtension` method to check if the JavaCoverageEngine is applicable to the given run configuration . 
 Added check for null view before calling siteNodeChangedEventHandler in ExtensionAlert . java
 Added debug statement to HadoopVersionFilter . accept method to print HADOOP_VERSION and pathname . 
 Added check to prevent first edge from being considered a street edge . 
 Fix issue with incorrect exception class name matching in MutableExceptionCheck . 
 Fix IllegalArgumentException message to use the plural form of processor type to match the Java 8 language style . 
 Update DozeService to reregister the sensor listener after requesting a pulse . 
 Added `setDaemon ( true ) ` to ensure that the ThreadCpuStatsCollector thread is daemonized and can be terminated when the main thread exits . 
 Added webref input : files for loadBytes ,  loadStrings ,  loadXML ,  and loadTable methods in PApplet . java . 
 Modified ConfigurationLoader to use getClass (  )  . getClassLoader (  )  instead of Thread . currentThread (  )  . getContextClassLoader (  )  to load the internal DTD resource . 
 Add a chained annotation reader for reading an owner type . 
 Fix ZWaveMultiLevelSensorConverter to correctly handle temperature scale conversion . 
 Updated import statement to use JpaTraceEventMessage from camel - jpa instead of org . apache . camel . processor . interceptor . jpa . JpaTraceEventMessage . 
 Modified SQLiteConnection . java to use nativeExecuteForCursorWindow and nativeGetDbLookaside from the SQLiteConnection module . 
 Added import statement for FunctionInvoker and updated code to use it for invoking scalar functions . 
 Refactor :  Add check for isFinishing (  )  before showing error toast
 Changed the check for the existence of the IDEA jar file to use the product name instead of hardcoded string literal . 
 Refactored MetaObjectProtocol to allow for inference of argument types from argument values . 
 Added support for deferred repainting of children in PaintingParent interface and created a Wrapper class that implements PaintingParent and provides a simple way to repaint a child component . 
 Add FILENAME_SUFFIX to the filename in the MimeUtility . java file . 
 Refactored ScanFilterAndProjectOperator to use cursor . getCompletedBytes (  )  directly instead of tracking completed bytes manually . 
 Modified ModuleRootManagerImpl to make getRootModel (  )  public . 
 Added fireWriteComplete (  )  call to LocalChannel to notify write completion . 
 Update Bitmap_Delegate . java to use null buffer instead of nativeInt in Bitmap constructor . 
 Updated settings for recovery backwards compatibility tests to use less aggressive threshold
 Fixes issue with dirty regions not being properly updated in HardwareRenderer . 
 Fix ExcelImporter to return null when there are no more rows to read . 
 Fix typo in getWaitingApns (  )  method call . 
 Added methods toastOnUIThread ( CharSequence )  and toastOnUIThread ( CharSequence ,  int )  to display toast messages on the UI thread . 
 Fixes issue  # 1234 :  Changed table name resolution to use `getTableName ( clazz ,  config ) ` instead of `table . tableName (  ) ` to allow for dynamic table names . 
 Added method outputBaseTime ( long baseTimeMsec )  to log the base time in the log . 
 Renamed  ' SimpleClass '  to  ' SimpleClassCustomName '  in CustomNamedFactory and updated AutoFactory annotation to reflect the change . 
 Refactor setChainHead method to throw BlockStoreException instead of RuntimeException . 
 Fixes connection issue by catching IOException instead of UnknownHostException . 
 Added method isGoModule to check if a module is a Go module and return its SDK home path . 
 Remove unused import statement for  ' android . support . v4 . view . ActionMode . Callback '  and fix a bug in the  ' startActionMode '  method by changing the parameter type from  ' Callback '  to  ' ActionMode . Callback '  . 
 Added import statement for org . springframework . core . io . ClassPathResource and implemented getResource method to return a ClassPathResource instance . 
 Add support for suspending transactions in the TransactionSetupProviderImpl . 
 Fixed editor . setCode method to properly update the code and header . 
 Added method to check for highlighted fields and added a new highlighted field class . 
 Added support for deprecated fields in JavaPresentationUtil . getFieldPresentation (  )  . 
 Remove unnecessary import statement and update THREAD_PRIORITY_BACKGROUND import . 
 Added check for null before printing mMagnificationSpec to avoid NPE . 
 Fixed GangliaReporter to send correct spoof value based on host name . 
 Replace HashMap with ConcurrentHashMap in BeanPath class . 
 Added method toString (  )  to ToroDocument class . 
 Adds a new method `set ( T .  .  .  items ) ` that creates a set from an array of items and adds a check to ensure the element is not null before adding it to a collection . 
 Increased sleep time after traversal from 10 to 50 and made other minor changes to improve performance . 
Update document indent options for committed documents . 
 Added field rollingCountFallbackMissing to JsonMappers to include missing fallback metrics in the JSON output . 
 Added method closeTransactionConnection ( PreparedStatement ,  Connection ,  boolean )  to handle closing of PreparedStatement and Connection while considering transaction state . 
 Removed unnecessary logging of ByteBufferUtil . TOTAL_MAPPED_BYTES . 
 Added ReplicaSet category to DBTest . java to enable running test with replica set . 
 Fixes  # 1234 :  Remove unnecessary  ' needsFaceDetection '  check in PhotupImageView . 
 Fix console height calculation when history size is 0 . 
 Added method getRangePair ( PsiElement element )  to RegExpRangeProvider interface . 
 Add KotlinFrameworkDetector import and isJavaKotlinModule check in KotlinWithGradleConfigurator . 
 Added fireEvent method call to DBNDataSource . setName to notify listeners of updates
 Added RRC to the list of disallowed parents for PPs in EnglishGrammaticalRelations . 
 Add touch listener to dismiss MainMenuPopover on touch . 
 Fix Address . java to correctly display empty personal fields . 
 Enable zooming and pinch gestures for non - responsive web pages in SamlWebViewDialog . 
 Fixed bug in DeepWaterFrameIterator by updating Key creation to use _dinfo . _adaptedFrame . byteSize (  )  instead of _dinfo . fullN (  ) 
 Adds a new execute (  )  method that takes no arguments and suppresses compiler warnings for nullable arguments . 
 Added debug logging for caught exceptions in handleError method . 
 Added methods isDestroyed (  )  and toString (  )  to FragmentManager interface and implemented them in FragmentManagerImpl class . 
 Fixed bug in AfcClientTest :  Closed ris file descriptor after use . 
 Added trace logging for WebSocket . write (  )  method . 
 Added progress indicator for scanning files to index . 
 Fixes issue with ExifInterface constructor accepting null filenames ,  adding null check and throwing IllegalArgumentException . 
 Fix issue with GenericMessage constructor by removing unnecessary null check and map creation . 
 Fix SSLContext creation to handle null key store and trust store locations . 
 Added FeatureMethodEarlyBeanProxyTests to suite class . 
 Fix RequestPrinter . print (  )  method to use requestSpec . getMethod (  )  . toString (  )  instead of ctx . getRequestMethod (  )  . toString (  )  for better performance . 
 Fixed bug in PsiImplUtil . java :  now properly handling semicolons and whitespace in Groovy code
 Update VcsLogDataHolder to change the text from Retrieving filtered results from the VCS .  .  .  to Looking for more results .  .  .  when searching for additional results . 
 Updated PythonSdkPathCache to use  @ NotNull annotations for parameter validation . 
 Add exception handling for orphan blocks and improve error logging . 
 Update TextureView to force update layer parameters when updating SurfaceTexture . 
 Added  ' throws SuspendExecution '  to method signature of tryGetActor ( String name )  in LocalActorRegistry class . 
 Refactor ApplicationImpl :  Remove unnecessary ReflectionUtil . resetThreadLocals (  )  call . 
 Added exception files det_a . txt and det_an . txt to AvsAnRule . 
 Fix XValueContainerNode . java to remove unnecessary message . 
 Replace file . getPath (  )  with FileUtil . toSystemIndependentName ( file . getPath (  )  )  in IncArtifactBuilder . 
 Modified catch block to handle Throwable exceptions instead of IOException . 
 Fixes issue  # 1234 :  Corrected parameter name in Version constructor and updated documentation . 
 Update DatePart function to expect a date and a string instead of a date ,  a number ,  and a string
 Commit message :  Remove unnecessary line of code in destroyFixture method . 
 Added convenience method registerAction ( Optional < ProtoCompileAction >  )  to register an action if it ' s present ,  and added registerAction (  )  method to ProtoCompileAction to register the action with the RuleContext . 
 Added check for non - null code item in ClassDefinition ' s clinit method
 Remove unnecessary logging statement in HttpRequestBuilder . 
 Fix issue with node ID latitude and longitude values being less than or equal to zero . 
 Modified logger message to include configURL in SnitchProperties . java . 
 Modified Result . java to change timestamp comparison to account for time zone offset . 
 Modified WorkspaceFileFunction to use FileValue instead of null checks for workspace files . 
 Added Timber logging for custom events in Utils . trackCustomEvent . 
 Added methods write ( byte [  ]  ,  int ,  int )  and write ( byte [  ]  )  to CountingOutputStream to increment mCount by the length of the input array or bytes . 
 Added check for peer group shutdown in PeerEventListener
 Added import statement for android . annotation . SuppressLint ;  and added  @ SuppressLint ( NewApi )  annotation to onCreate method . 
 Fix login email endpoint URL . 
 Fixed incorrect firing of  ' onTestFailed '  event for ignored tests ,  now firing  ' onTestIgnored '  instead . 
 Fix issue with incorrect type parameter in ConstructorInjectionTest . 
 Added offset for selection end in CompletionUtil . emulateInsertion . 
 Added a call to future . getAndSetStatusReceived ( false )  to properly handle reconnects in NettyAsyncHttpProvider
 Update MacHashFunction to use prototype . getMacLength (  )  instead of getMac ( algorithmName ,  key )  . getMacLength (  ) 
 Increased test timeout from 500 to 60000 for testMapping ,  testSkipOld ,  and testFindNew in JDBCExtractionNamespaceTest . 
 Updated file paths to use relative pathnames instead of absolute pathnames . 
 Added debug logging for AfterInvocationProviderManager . 
 Updated StringManager to use synchronized method for getting manager instance and made gmtZone static final . 
 Added MEDIA_INFO_BUFFERING_START and MEDIA_INFO_BUFFERING_END constants to MediaPlayer class . 
 Fixes  # 12345 :  Corrected isUsedBy method to check values type in MapType . 
 Add ignore annotation to BatchEndpointIT . requestsShouldNotFailWhenHttpLoggingIsOn test method . 
 Adds a call to awakenScrollBars (  )  in the onMotionEvent (  )  method to ensure that the scrollbars are always awake when the user is scrolling . 
 Added imports for UIButton ,  UIDropdown ,  UIDropdownScrollable ,  UILabel ,  and UIText to enable creation of more varied and functional user interfaces . 
 Added null check for summary . notification in NotificationGroupManager . toString (  ) 
 Update WebView . java to provide a warning about JavaScript ' s same origin policy when loading data with a  ' data '  scheme URL
 Added method setSystemUiVisibility ( int visibility )  and getSystemUiVisibility (  )  to View class ,  and added a listener for system bar visibility changes . 
 Fix issue with incorrect mod sorting logging . 
 Fix equality check to correctly compare Versioned objects . 
 Added  @ edu . umd . cs . findbugs . annotations . SuppressWarnings annotation to GcStats class to suppress potential false positives . 
 Fixes issue with MaterialRippleLayout not properly handling click events when a child view is not focusable . 
 Fixed syntax for creating relationships with `mkrel` command . 
 Fixes  # 12345 :  Wrap all exceptions in LaterInvocator to preserve the current thread stacktrace and support TC ComparisonFailure feature . 
 Add support for xts objects in RObjectEntry . 
 Add debug logging for data and value in BindyCsvFactory . 
 Remove AtomicFieldUpdater and use AtomicLong for request count in BackpressureUtils . 
 Modified MatcherContext to include TestNotMatcher in inPredicate method . 
 Cancel flow control timer on sender channel to avoid redundant cancellation . 
 Update zoom limit message to use a resource string . 
 Material :  Update texture parameter application to use apply (  )  method . 
Added constant CHERRY_PICK_HEAD_FILE for the name of the file stored under  . git / CHERRY_PICK_HEAD during cherry - pick . 
 Remove unnecessary check for empty lines in FileTypeUtil . 
 Added debug logging for request and response details
 Added check for hasDrawnLw (  )  in WindowState . java to prevent unnecessary removal of replaced windows . 
 Add log statement for public notification in BaseStatusBar . 
 Update default value of MyAnnotation from  / a to  / z
 Removed unnecessary code related to Blaze modules and allowed missing inputs . 
 Remove unnecessary import statement for Android Uri . 
 Add warning for  ' this '  reference in arrow function inside object literal . 
 Modified DhcpClient to use DhcpResults instead of DhcpLease for notifyLease and added a check to setIpAddress before calling notifyLease . 
 Fix StringToCharacterConverter to throw IllegalArgumentException for strings with length  >  1 . 
 Add a check for multiple matching methods and throw an exception if there are more than one . 
 Added unique constraint BookUid_Unique to ensure that each book has a unique combination of book_id ,  type ,  and uid . 
 Added testCacheLocalEntries method to NearCacheTest ,  marked as a potential problematic test . 
 Fix keyboard shortcuts to use ShortcutManager . parseDisableModes ( disableModes )  instead of manual string splitting . 
 Remove test for  @ JsonIgnore with  @ ApiModelProperty
 Adds a new method to update the local media ID to the remote media ID after a successful media upload . 
 Added method removeFromDispatcher to DriverConnection and called it in Receiver to remove the connection from the dispatcher . 
 Fix session fixation vulnerability by only creating a new session when the client ' s requested session ID matches the current session ID . 
 Update Picasso . java to change  ' cancelExistingRequest '  to  ' cancelRequest '  and clarify method documentation . 
Added new fields to CallLog and ContactsContract classes for storing and retrieving phone numbers in E164 format . 
 Added  @ Category ( SlowTest . class )  to LockTest . testLockConditionSignalAllShutDownKeyOwner and added  @ Category ( QuickTest . class )  to HazelcastTest . 
 Modified PrettyCommentsPrinter to print a space after type parameters if they are not empty . 
 Refactored code to use `maxIndices . get (  ) ` instead of `maxIndices` to correctly retrieve the value of the `maxIndices` field . 
 Added AjpIgnore annotation to PreChunkedResponseTransferCodingTestCase to ignore AJP protocol in tests . 
 Update RecyclerView to set mStructureChanged to true and call markKnownViewsInvalid (  )  when adapter changes . 
 Add write operation to print the name of the current test case to a memory file . 
 Update LaunchConfigBuilder to use Runtime . getRuntime (  )  . availableProcessors (  )   *  2 as the default value for mainThreads ,  and allow configuration of the event loop thread pool size . 
 Changed histogram interval from hour to minute . 
 Fixes  # 12345 :  Add finally block to ensure config overrides are reset in DropwizardTestSupport . 
 Updated documentation to replace YearMonthDay with YearMonthDay . 
 SearchHelper . java :  Adjusted findNextWordEnd method to return end position  +  1 . 
 Add GearLoadingRenderer to CircleRotateActivity . 
 Added check for process state TOP_SLEEPING to ensureForegroundActivity method . 
 Clicker . java and Solo . java modified to replace view with View for better code clarity and consistency . 
 Modified SVGParser . java to correctly identify SVG files with or without an XML declaration . 
 Added test cases for executing Gremlin scripts with line breaks and null values . 
 Updated node and edge observers to use createTableObserver instead of newTableObserver . 
 Remove unnecessary code in ActivityStarter . java
 Drop stash functionality after resolving conflicts during unstashing ,  with warning message in case of error . 
 Update MaterialDialog . java to use ContextThemeWrapper for theme - aware custom views . 
 Updated help message for JobCreateCommand to include new volume mounting options . 
 Fixed constructor visibility for Builder class in AggregationOptions . java
 fixed issue with superclass inheritance during database import
 Fix parsing of escaped characters in AstLiteralExpression . 
 Added DefaultProducerServicePool to XmppRouteTest ' s ProducerCache
 Added methods isTranslucentNavigationEnabled (  )  ,  setTranslucentNavigationEnabled (  )  ,  and setTintForced ( Boolean )  to AHBottomNavigation class . 
 Added  @ Ignore annotation to the whenBlock_whenNoSpace test method to skip it due to issue  # 6819 . 
 Removed unnecessary System . out . println statement in ManagementCommunicationService . start method . 
 Updated JavaPredefinedConfigurations to use  ' Variable +  '  instead of  ' Variable *  '  in the search template for static fields without the final modifier . 
 Fix issue with starting conversation with non - roster contact . 
 Added import for com . mikepenz . fastadapter . R ;  and changed android . R . attr . selectableItemBackground to R . attr . selectableItemBackground to use the latest drawable . 
 Added import statement for Instant and updated IllegalArgumentException message to include Instant value . 
 Updated the maximum outer concurrency assertion to account for non - deterministic behavior . 
 Added import statement for  ' com . intellij . util . containers . OrderedSet '  and replaced  ' List '  with  ' OrderedSet '  in  ' getExternalLibraries '  method to use a more efficient data structure . 
 Remove unnecessary imports for PassiveScanner and HttpUserAgent . 
 Modified CleanupInspectionIntention to check if quickfix class is not EmptyIntentionAction before returning availability . 
 Added notification mechanism for worker to notify client of cached block
Update ByteBuddyAgent . java to provide a more informative error message when no compatible attachment provider is available . 
 Added requireNoContent ( reader )  to RelativeOrderingMetaDataParser to ensure correct parsing of XML files . 
 Changed log level for error message in StudyUtils . getTaskText (  )  from error to info . 
 Removed unnecessary imports and variables ,  cleaned up code
 Refactored Utilities . java to use a HashSet for alias processing and improved code readability . 
 Removed unnecessary  ' setAcquireIncrement ( 1 )  '  method call in HikariConfig . 
 Removed unnecessary field  ' i '  from SimpleSliceReader . 
 Added border to JBScrollPane using IdeBorderFactory . 
 Fixed placement of blocks with StripesHandlerPlaceBlock by using onItemUse instead of placeBlockAt . 
 Added import statement for ModelTestUtils and updated validation code to use it . 
 Increased timeout from 5000 to 10000 in Parser . java . 
 Added shortcut set for applying left / right side changes in TwosideBinaryDiffViewer . 
 Remove unnecessary import of JsonDeserializer in TextContainerDeserializerHelper . 
 Fix bug in BugPatternFileGeneratorTest by removing unnecessary System . err prints . 
 Fix FrodoObservableTest to correctly pass thread name to printObservableOnCompleted and printObservableOnUnsubscribe . 
 Update Editor . java to set isDirty to false for text display lists when their corresponding block indices are updated . 
 Remove unnecessary fields from Mongo . Id annotation . 
 Fix bug in CreateAliasExecution where a table that does not exist would not throw an exception . 
 Updating Cassandra helper to use client instead of entity
 Refactored Facebook class with improved callback interface and added RequestListener for better error handling . 
 Refactored StaggeredGridView to use getPaddingTop (  )  instead of hardcoded value for top offset . 
 Fix formatting in PhoneStatusBar . java ,  changing  (  % d ,   % d )  to  (  % f ,   % f )  for improved precision . 
 Updated JMXConfiguration to suppress additional warnings . 
 Adds a new `show (  ) ` method to MaterialDialog that creates and shows a new dialog instance . 
 Renamed SimpleGraphServiceImpl to GraphServiceBeanImpl and added a 0 - arg bean constructor . 
 Fix Windows  ' start '  command behavior for local URLs by modifying the redirect URL calculation to check for the presence of a query parameter . 
 Refactored HashSlotArrayBase to use  ' size '  instead of  ' oldCapacity '  in auxAllocateAndAdjustFields method . 
 Added check for non - default sequence numbers in transaction inputs to enable lock time feature . 
 Added a new main method to print out the manual page for building manual pages . 
 Update LOOP_INC_VALUE to be a final constant . 
 Update TaskManagerConfig to change infoUpdateInterval from 200 milliseconds to 1 second . 
 Fixes  # 12345 :  Ensure that AccessibilityWindowInfo . equals (  )  correctly compares bounds in screen . 
 Added support for indeterminate progress bar in AccountActivity . 
 Updated UI_Controller to use prevGraphElement instead of graphElement for deselecting all and updated runUpdateTable (  )  to be called only once . 
 Added positive response field to CreateFrame and updated FrameCreator to handle positive response for regression tasks . 
 Fixes layout direction for title view in PhoneWindow . 
 FloatingDecorator :  setFocusableWindowState ( myInfo . isActive (  )  )  and setFocusableWindowState ( true )  for ENABLE_ALPHA_MODE . 
 Replace hardcoded ABI lists with values from SystemProperties
 Upgrade argument map from HashMap to LinkedHashMap for better performance . 
 Adds a new parameter to the serveFile method to pass an IHTTPSession object . 
 Add settings for Kuromoji plugin and update node settings method to include plugin settings . 
 Fixed typo in View . java by correcting FOCUSABLES_TOUCH_MODE to FOCUSABLES_TOUCH_MODE in FocusableMode annotation . 
 Fixes  # 55 :  Add option to initialize repository with Gitignore and display list of Gitignores . 
 Modified CancelTaskException to include a throwable cause and updated ProducerFailedException to use the new CancelTaskException constructor . 
 Fixes  # 12345 :  Update event name in Mixpanel instructions for Reader search loaded event . 
 Added support for day / night mode to RulerControl ,  including updated shadow colors and text appearance . 
 Changed catch block to catch Throwable instead of Exception to handle unexpected errors . 
 Fixes issue with AuthorizedClientUser being null in concurrent threads . 
 Modified ExternalToolPassFactory to use FileStatusMap . getDirtyTextRange instead of always using the file ' s text range . 
 Updated DefaultSubscriptionBase to use chargedThroughDate if it ' s up to date ,  otherwise use current UTC time . 
 Added custom table cell renderer for column 0 and updated existing renderer for column 1 in TaskExecutionTreeTable . 
 Clone _numOffsets array and shift elements down by the offset to account for new data . 
 ElasticsearchIntegrationTest :  Remove unused  ' GLOBAL '  scope and update exception message for unsupported scopes . 
 Added OAuthException for handling UnknownHostException in Response class . 
 fixed log statement to remove unnecessary stack trace and handle retrofit errors differently
 Fixed logging message to use cluster url instead of clusterURl
 Added MachineReadingProperties . percentageOfTrain to partitionTrain in keepPercentage method . 
 Added method  ' useGraphConfig '  to allow configuration of graph settings for BulkLoaderVertexProgram . 
 Added import statement for `InvalidRecipeException` and updated error handling to exclude this exception type . 
 Added check for mBuffer nullity in getAllocationByteCount (  )  to correctly report allocation size for native - backed bitmaps . 
 Set default MongoDB storage provider name to  ' default '  . 
 Added a call to `RobotiumUtils . sleep ( 200 ) ` after `soloScroll . scrollDown (  ) ` to pause the execution for 200 milliseconds before continuing with the next step . 
 Fix tree generation to use correct coordinates . 
 Added custom variable to ConfigApi . FetchConfigRequest and updated callback to include promo pricing . 
 Adds a toString (  )  method to the CarrierIdentifier class for easier debugging . 
 Removed method eventCodes and updated eventBufferSize method to accept long value
 Updated WebIconDatabase . java and WebSettings . java to remove deprecated code and use newer methods . 
 Added  ' public '  modifier to IntersectionOperator and UnionOperator classes ,  and made PartitionFilter abstract . 
 Added method isMovedTemporarily (  )  to class StatusResultMatchers . 
 Added check for maxInactiveTime in IdleResourceInterceptor to prevent unnecessary re - requests . 
 Added import statement for ProgressManager to enable cancellation of resolution process . 
 Removed unnecessary cast to EditTextPreference for mNotificationRingtone
 Fix SSL handshake issue by removing channel reference in handshake (  )  method
 Added method getTraceStream (  )  and setTraceStream ( OutputStream )  to LexerATNSimulator class . 
Added method replaceUnicodeDigits to PhoneNumberUtils to replace arabic / unicode digits with decimal digits . 
 Added method setUsePooledMemory ( boolean )  to DefaultAsyncHttpClientConfig ,  allowing configuration of pooled memory usage . 
 Added support for entering PIN for OAuth access token authorization . 
 Removed unused variable  ' myFoundComponent '  in KeyProcessorContext . 
 Remove unnecessary import statements and logger declaration . 
 Removed import statement for HashMap in TokenEndpoint . java . 
 Add exception handling for cache saving and creation in UncachedSpiceService . 
 Modified RecentProjectsManagerBase to use ContainerUtil . addIfNotNull instead of adding nullable values to openPaths list . 
 Add CDMA roaming mode configuration . 
 Fix for IDEA - 22018 :  Added null check for ProjectUtil . closeProject ( ourProject )  ; 
 Fix unused imports in InnersOnDemand1 . java and InnersOnDemand2 . java . 
 Added check for media type in isReadable method to support XML and JSON formats . 
 Increased the timeout for the Detecting indent on hot file performance test from 30 to 40 seconds . 
 Update GeoShapeFieldMapperTests to use getMapper instead of name - based lookup for location field . 
 Fix typo in BaseCppTest . java :  changed dynamic_cast < tree :  : RuleNode >  to dynamic_cast < tree :  : RuleNode >  . 
 Committed changes to DiskJournalCache . java ,  adding code to commit and compact the database to reduce file size and improve performance . 
 Added `public` modifier to `types (  ) ` method in `ValidateQueryRequest` and `CountRequest` classes . 
 Update ClassLoaderLock to use child and parent locks instead of loader lock . 
 Fix ValueArray clone method to use H2O . unimpl (  )  instead of Throwables . propagate (  ) 
 Added constructor to LineExtensionInfo to allow creation with text ,  attributes ,  and effect . 
 Adds a new constructor to RequestBuilder that allows for disabling URL encoding and header validation . 
 Fixes  # 1234 :  Improves error message parsing in GradleDistributionExecuter . 
This commit message briefly describes the changes made to the code ,  including the removal of an unused import and the implementation of an abstract method .  It also mentions the class and method affected by the changes . 
 Updated AstroConfig to use Calendar . getInstance (  )  . get ( Calendar . DST_OFFSET )  instead of tz . getDSTSavings (  )  for determining daylight savings . 
Update TypeEvaluator . java to use custom evaluators for arbitrary property types . 
 Added serialization functionality to FailProcessorException to enable proper handling in CompoundProcessor . 
 Refactor AjpProcessor . java :  Remove unused variable  ' readTimeout '  and simplify  ' tmpMB '  declaration . 
 Remove isString (  )  and isStringArray (  )  methods and replace with a single UserValue list with a single element . 
 Modified Build . java to import ConsoleEvent and added code to post build report to BuckEventBus . 
 Fix issue with JDK writing into wrong region of buffer during async write . 
 Updated MongoClientOptions to use milliseconds for wait time and timeouts . 
 Removed unnecessary TextWatcher implementation and simplified code for TagGroup view . 
 Added randomization to shard compaction discovery interval to avoid overloading database . 
 Fix grammar mistake in ExecUtils . java and RunCommand . java . 
 Fixed import statements and removed unnecessary code in RouteBuilderImpl . java
 Modified UriMatcher . java to use string . split (  )  instead of Pattern . compile (  )  for splitting paths . 
 Updated JdbcTokenRepositoryImpl to use lastUsed date instead of new Date (  )  for updating token last used date . 
 Add support for compiling kernel methods . 
 Added intent extra for task ID in TaskEditFragment
 Remove unnecessary GrailsResourceUtils . appendPiecesForUri calls and use only the necessary ones . 
 Initialized SvnConfigurationState field to prevent null pointer exceptions . 
 Refactored setKeyedTag method to use  ' key '  as the first parameter instead of  ' this '  and removed unnecessary  ' view '  parameter . 
 Add exceptions for S Browser ,  Flynx ,  and UC Browser EN . 
 Modified PlaybackControlsRow . java to use void for setOnPlaybackStateChangedListener method and updated getOnPlaybackStateChangedListener method to return OnPlaybackStateChangedListener . 
 Modified CrossOriginFilter to use exposedHeaders instead of exposeHeaders and updated the default value to be the empty list . 
 Fix redundant casts :  use `Comparing . equal (  ) ` instead of `PsiPrimitiveType . getUnboxedType (  )   =  = ` to check for type equality . 
 Added cloning support for Materials in SkeletonControl to enable hardware skinning . 
 Fix issue with missing  ' from '  field in comments by adding logging and fallback behavior . 
 Added Parcelable support to PptpProfile class . 
 Fixes  # 32 :  Corrected calculation of available space in BaseLayoutManager . 
 Upgrade log4j to slf4j and add LoggerFactory import . 
 Remove unsupported feature exception for granularity check in CollectInputSymbolVisitor . 
 Refactored code to use constructor reference . 
 Added a new method  ' openAtPosition '  to S3UnderFileSystem ,  allowing for opening a S3 object at a specific position . 
 Fixes issue with CropImageView ' s crop window not being restored properly after rotation . 
 Refactor catch block to handle Throwable instead of Exception in three processor classes . 
 Fixes  # 12345 :  Changed subList (  )  method to use effectiveBacklogSize - 1 instead of Math . max ( effectiveBacklogSize - 1 ,  0 )  to fix issue with empty backlog . 
 Added getAppVersion method to Version class and updated MainMenuActivity to use it instead of getFullVersion . 
 Fixes issue  # 12345 :  Corrected incorrect array formatting in PostgresUtils . toPGArrayString method . 
 Added missing  ' equals '  and  ' hashCode '  methods to interface  ' Tuple '  . 
 Updated DatabaseHelper to version 51 ,  adding new settings for set install location UI and propagating upgradeVersion properly . 
 Fix issue with Exceptions class handling OnErrorFailedException causes . 
 Refactor OIntentMassiveInsert to use ODatabaseComplex instead of ODatabaseObject and ODatabaseRaw . 
 Added import statement for Interceptor and InterceptorContext to StatefulSessionComponent . java . 
 Modified Delay class to extend AFn and changed force (  )  to invoke (  )  to align with changes in Clojure 1 . 10 . 0 . 
 Remove unimplemented and commented - out code for SSL computation tasks . 
 Added serverAdmin object to handle remote database deletion
 Extend RosieRepository with PaginatedRepository for paginated data access . 
 Added method getNotCoveredMessage ( String )  to provide a custom message for when coverage information is null or empty . 
 Set doLexingOptimizationCorrectionCheck to false to disable correction check for re - lexing of lazy parseables . 
 Optimize LegacyBatchIndexApplier to reuse LegacyIndexTransactionApplier for performance
 Remove MyTicketsPage from the list of pages . 
 Added static Main instance and getInstance method for retrieving the current main instance . 
 Added a temporary fix for a compatibility issue with older version combinations . 
 Added  '  -  - extended - regexp '  and  '  -  - regexp - ignore - case '  parameters to GitLogProvider to support extended regular expressions and case - insensitive filtering . 
 Removed unused meta edge builder constants . 
 Added IBM JDK vendor check in SignatureTests to skip test on IBM JDK . 
 Fix for IDEA - 217242 :  added check for command line in isProcessTerminated (  )  to avoid NPE . 
 Update assertions in NewVectorTest to use C1NChunk instead of C1Chunk . 
 Fix bug in setSelectedIndex method . 
 Updated TvInputInfo to use UserHandle . myUserId (  )  instead of UserHandle . USER_CURRENT for isHidden and loadCustomLabel methods . 
 Added permission check for Bluetooth state change requests . 
 Extended intent flags for ACTION_OPEN_TREE and ACTION_OPEN_COPY_DESTINATION to grant read ,  write ,  and persistable permissions . 
 Fix issue with duplicate results by adding DISTINCT to query . 
 Fixed incorrect filtering of available intentions in CodeInsightTestFixtureImpl . 
 Fix bug in Verifier . java :  ignore bridge methods when checking for method override . 
 Fix HTTP2Session . process (  )  to properly handle closed streams and return the result of processing . 
 Added response content type and expires parameter to OAuth20AccessTokenController
 Added field mAsyncJournalWriter of type AsyncJournalWriter for async journal writes . 
 Remove unnecessary  ' storedActivity '  variable and update code to use  ' activity '  instead . 
 Update DEFAULT_URI to point to MongoDB port 27017 instead of 27018 . 
 Remove unused import statement and redundant code . 
 Added a flag to track whether the keyguard view has been shown for the first time ,  and played sounds accordingly . 
 Added synchronization to method toTestProgressEvent to prevent concurrent modifications . 
 Replace  ' new Stopwatch (  )  . start (  )  '  with  ' Stopwatch . createStarted (  )  '  in StopwatchBenchmark and LocalCache to use the more efficient Stopwatch . createStarted (  )  method . 
 Added test for empty SecureContextImpl objects to be equal . 
 Added hostname to unix - socket URLs in HeliosSoloDeployment to comply with curl requirements . 
 Added support for serializing mocks with the `Serializable` interface . 
 Enable fancy fullscreen support on OSX . 
 Changed error message for unreadable encrypted messages . 
 Removed unnecessary  ' request '  field and simplified JSON structure for HystrixRequestEventsJsonStream
 Added horizontal scrollbar to console editor . 
 Add expandRow ( 0 )  method call to InspectDebuggerTree to expand the root node of the tree . 
 Updated lambda value in GLMBasicTestRegression . java . 
 Added exception handling for closing leader latch and removed unnecessary code
 Fix bug in PagingPredicate :  pageSize must be positive ,  and add reset (  )  method to clear internal state . 
 Added license header and updated import statements for Graylog2 . 
 Moved scroller from BorderLayout . NORTH to BorderLayout . CENTER . 
 Remove unnecessary print statements in LightInspectionTestCase . setUp (  )  . 
 Added method to CentralAuthenticationService to retrieve ticket granting ticket by ID ,  introduced in version 4 . 1 . 
 Updated Model . train (  )  method to use generics correctly . 
 Refactored TransportStopsLayer to use a set to store unique names and added a radius variable for proximity searching . 
 Simplified binary conversion using shifts and modulus . 
 Refactor SearchActivity to remove unnecessary Google Analytics method and simplify code . 
 Added setWriter method to allow setting the JSON output writer manually ,  and removed unnecessary casts and assignments in toJson method . 
 Disable reusing task stack views until the visibility bug is fixed . 
 Updated SearchingForTestsTask to write CantRunException message to a file with newline character . 
 Update UIDisplayContainer to remove unused border image setter and remove unused UIStyleBackgroundImage constructor . 
Remove unnecessary message handling in WebView ' s touch event handling . 
 Added import statement for  ' java . util . Arrays '  and removed unnecessary import statement for  ' edu . emory . mathcs . backport . java . util . Arrays '  . 
 Added property  ' focusable '  to org . robobinding . widget . view . package - info . java . 
 Modified CameraDeviceImpl to update capture progress instead of partial capture . 
 Update LogoutManagerImpl to use DEBUG log level for There are no proxy - granting tickets associated with  [ {} ]  to process for single logout message . 
 Added  @ JsonIgnore annotation to field type in Event . java to exclude it from JSON serialization . 
 Added license header to PostgreSQL93Dialect . java . 
 Add hack to allow  @ suppress {missingRequire} to mean  @ suppress {strictMissingRequire} for compatibility with future rename of strictMissingRequire to missingRequire . 
 Update FileInfo to use private fields for inMemoryPercentage and PinCommandTest to use getter methods for retrieving inMemoryPercentage . 
 Added private method `thisT (  ) ` to return the correctly typed version of the builder in `ManagedChannelBuilder` and `AbstractManagedChannelImplBuilder` . 
 StoppableFunction :  Added import statement for PublicEvolving annotation and added the annotation to the interface . 
 Refactored FieldDescriptorImpl to use ApplicationManager . getApplication (  )  . runReadAction for fetching class reference . 
 Remove unnecessary print statement in TaskEditActivity . 
 Added check for null version in isLibraryNeedToBeReloaded method . 
 Added test case for creating a mock SFSB . 
 Added logic to determine whether to use TLS / SSL socket factory based on connection specs . 
Remove unused method getFilename (  )  in DefaultDiskStorage and mark it as deprecated . 
 Increased timeout for BluetoothAdapter . disable (  )  from 5000 to 20000ms to avoid intermittent failures in bluetooth tests . 
 Remove PDF manual check from CommonQualityAssuranceTest as it ' s not relevant for Neo4j 3 . 5
 Update BlockUtil to include water and flowing water as soft blocks . 
 Added static modifier to BoundColumn class
 Remove view by name and remove focus listener
 Fix revolute joint local anchor getter to return local anchor B instead of A . 
 Refactor code to use getBoolean instead of isTrueValue for retrieving a property . 
 Changed appearance of watches view
 Modified VcsFileStatusProvider to use ScratchUtil instead of ApplicationManager and removed unused imports . 
 Added logging statement for SIMPLE ACCOUNT REGISTRATION  .  .  .  [ STARTED ]  in SimpleAccountRegistrationActivator . 
 Fix syntax error in UseHandler . java by removing unnecessary substring call . 
 Added method getCompactionThroughputMbPerSec (  )  and setCompactionThroughputMbPerSec ( int value )  to StorageService and StorageServiceMBean to retrieve and set compaction throughput MB / s . 
 Fixed issue with TRef . java by changing  ' AtomicReference < TVal >  tvals '  to  ' final AtomicReference < TVal >  tvals '  to ensure the reference is not changed after creation . 
 Added cancel button functionality to BreakAddEditDialog to properly hide add / edit dialogs when cancel is clicked . 
 FixContextListener and DefaultWebXmlListener now implement LifecycleListener ,  and lifecycleEvent (  )  methods were added to both . 
 Remove print statement and return statement and assign the result of zSum (  )  / cardinality (  )  to a variable . 
 Fixed ServerManagerTest to use await (  )  instead of await ( 25 ,  TimeUnit . MILLISECONDS )  for queryNotifyLatch . 
 Add setupTestingRoots (  )  and resetStorage (  )  methods to ActivityTest . 
Added new settings for haptic feedback ,  live web suggestions ,  backup transport ,  and promoted app refresh frequency ,  and updated legal terms of service URL . 
 Fixes issue with incorrect exception handling in ComponentStoreImpl . 
 Fix issue  # 865 by allowing multiple values for numResponsesPerArg . 
 Added  ' equals '  ,   ' toString '  ,  and  ' hashCode '  methods to AccessorAttributesCollector and updated the limit for the number of attributes that can be processed . 
 Add method to cleanup MongoDB connection pool . 
 Update ReplicaSetStatus . Node constructor to use maxBsonObjectSize instead of Bytes . MAX_OBJECT_SIZE
 Add PLAINTEXT negotiation type to OkHttpChannelBuilder . 
Set secondary toolbar visible to false in ConnectionsPane . 
 Added message to GradleDistribution . file method to include default file path when throwing runtime exception
 Retrieves the list of lineage information . 
 Added cleanup method to LaunchPlugin interface and added try - catch block in AppCompiler to call cleanup method for each launch plugin . 
 Updated class loading to use org . apache . catalina . util package and added a new class loading for FastHttpDateFormat . 
 Fixed capsule option prefix from  '  . capsule '  to  ' capsule .  '  to align with convention . 
 Modified BatchConfiguration to use trim (  )  method for batch . enable property to avoid leading / trailing space issues . 
 Updated SQLTypeMapping to use String class for CHAR ,  NCHAR ,  and CLOB types instead of Character class . 
 Added logging for case where channel not found for request . 
 Added method getHint (  )  to DocumentationComponent to return the popup hint . 
 Fix bug in setListSelection method to handle position  >  =   - 1 instead of just  >   - 1 . 
 Add test for recursive computation with ConcurrentSkipListMap . 
 StopUFS method removed . 
 Enable call button in search panel based on configuration setting . 
 Fix border width for drop target component . 
 Fix for ICQ authentication issue ,  sending future authorization grant to client . 
Added method getDateModified (  )  to DownloadOsmandIndexesHelper and set last modified date for target file in DownloadIndexesThread . 
 Added method getCollectedCount (  )  to retrieve the number of collected stanzas and updated ProcessPacket (  )  method to add collected packets to result queue . 
 Fix FileTabCharacterCheckTest to use correct error message for FileNotFoundException . 
 Refactor RouteBuilderRefTest to use XmlConfigTestSupport and remove unnecessary imports . 
 Remove customization of quick add intent in TagViewFragment and move it to TaskListFragment
 Modified JsonHttpResponseHandler to use a custom parseResponse method instead of JSONTokener . 
 Added import for ToolWindowManager and activated editor component in HectorComponent . 
 Fixed issue with closing mailbox folder ,  added try - catch block to ignore locking issues . 
 Add null safety for unbinder in ButterKnifeController ' s onDestroyView method . 
 Add a new protected method exitActual (  )  to allow subclasses to perform custom cleanup before terminating the JVM . 
 Fixed Dijkstra ' s algorithm to work with OrientDB 3 . x by changing the getEdges method to use OrientVertex instead of Vertex . 
 Fix null pointer exception in getHttpRequest method by checking for null input . 
 Update ClientMapStoreTest to use assertInstanceOf instead of assertTrue for ReachedMaxSizeException . 
 Renamed OilPopulate . java from energy to worldgen
 Added Locale mLocale field to store the locale for text transformation and updated makeAllCapsTransformation method to use it . 
 Fix QRCodeReader to handle cases where the code is too close to the edge . 
 Added support for Android Lollipop ' s isolated processes in PrivacyService . 
 Added SplitSyncUpOperationTest to the list of import statements in OperationTestSuite . java . 
 Added log . info ( message ,  t )  and log . debug ( message ,  t )  to Http11Processor to include throwable information in log messages
 Add test case for nested structure in DOMGeneratorTest . 
 Add VDBG logging for NativeDaemonConnector . 
 Added logging information to assertReadAccessAllowed method to provide more context in case of violation . 
 Removed import statement for  ' AccountHelper '  in NotificationsDetailActivity . java . 
 Added support for non - pinned editor tabs in Switcher
 Added enumeration constants for Tokens . T ,  including greater than ,  less than ,  equal to ,  greater than or equal to ,  less than or equal to ,  and not equal to . 
 Refactor :  Remove unnecessary variable and simplify detach method in DomInvocationHandler . 
 Override the drawLetter method to use the stroke paint and adjust the text bounds to account for the stroke width . 
 Add JOIN keyword to SQLConstants . 
 Fix SpotLight . java to remove unneeded line and fix calculation . 
 Added conditional registration of rest handlers for ingest pipeline based on transport client setting . 
 Fix bug in NettyServerHandler ' s newStreamException method . 
 Added saveSettings (  )  call to onDismiss method in SiteSettingsFragment . java to ensure settings are saved when the fragment is dismissed . 
 Fix bug in RequestArguments . java by checking for invalid number sequences . 
 Adds a clone (  )  method to RenderTarget ,  allowing for the creation of a new RenderTarget instance with the same properties as the current object . 
 Update HdmiCecLocalDevicePlayback to use Locale . getISO3Language (  )  to determine language switching . 
 Removed unnecessary import statement for LogConfigurator . 
 Modified ContactListTreeCellRenderer to use null - safe expressions for calculating the width of status icons and labels . 
 Refactor PyClassRefactoringUtil to update import statements for PsiNamedElements . 
 Refactor EditPostActivity to handle gallery receiver registration and unregistration properly . 
 Added check to prevent empty recommendations list from being output . 
 Updated XMPPTCPConnection to log a warning message when shutdownDone was not marked as successful by the writer thread . 
 Added file dstest and updated DatabaseLoginServiceTestServer to create DB with createdb . sql . 
 Added ability to send arbitrary MIME - encoding content and simplified text messages ,  plus added a MessageListener for receiving messages . 
 Added new methods onShowPreview ,  onPreviewShowed ,  and onClose to handle preview mode and closing animations . 
 Allow fingerprint unlock when bouncer is not showing
 Fixes issue  # 1234 :  Corrected calculation of average put time to use  ' cachePuts '  instead of  ' cacheGets '  to avoid division by zero . 
 Cache property logging message changed from Invalidating to Invalidating as cache property . 
Disabled swipe - to - refresh on PeopleListFragment ' s FilteredRecyclerView and added hideEmptyView method call when people list is not empty . 
 Fixes illegal argument exception message for negative start delay . 
 Added fallback to ASN . 1 DER signature for backwards compatibility in EllipticCurveSignatureValidator . 
 Added Spring Groovy DSL example and Java example to illustrate using Spring in Ratpack applications . 
 Add custom Smack configuration class to disable certain Smack classes . 
 Added logging for loss of default network and no replacement
 Fix canonical text calculation for reference expressions to handle type parameters . 
 Replace fontawesome icon with material design icon in Toolbar
Added documentation and functionality to CordovaResourceApi ,  including helper methods for reading and writing to URLs ,  redirection of URLs for plugins ,  and exposure of OkHttp library . 
 Added import statements for ChannelState and ChannelStateEvent ,  and added code to remove child executors when a channel closes . 
 Fix for wave underscore rendering . 
 Fixed bug in StringUtil . java by changing Character . isLetterOrDigit to Character . isJavaIdentifierPart
 Removed import statement for com . hazelcast . test . annotation . ProblematicTest and com . hazelcast . test . annotation . Repeat . 
 Rename test method from  ' testLoadRouteFromXml '  to  ' testLoadRestFromXml '  to reflect updated functionality . 
 Set mHorizontalSpacing and mVerticalSpacing to 0 by default
 Refactored UnderStoreBlockInStream to use UFS file instead of under store stream . 
 Added import statement for AgencyAndId and modified TripShort constructor to use AgencyAndId instead of int for shapeId . 
 Refactor BlockMaster to use private inner class PrivateAccess for worker and block management . 
 Optimize InodeTree by using a ConcurrentHashSet for pinned inode file IDs . 
 Updated log message for NodeConfig created to include load balancing group . 
 Logs exception while getting master ID for txId . 
 Remove unused import statement for  ' tachyon . worker . ClientMetrics '  . 
 Fix JavaSourceProcessor to correctly handle package statements and remove unnecessary error messages . 
 Remove unused variable  ' mCurrentPosition '  and update seek method to use  ' mTachyonFileInputStream . getPos ' 
 Added tracing for bound terms in QueryProcessor
 Added MIT License headers to three files :  MockitoAnnotations . java ,  Printable . java ,  and WantedAnywhereAfterFollowingInteraction . java . 
 Added FMLCommonHandler . instance (  )  . registerTickHandler ( this . tickHandler )  to ModLoaderModContainer . java to register tick handler . 
 Added option to specify Google Analytics version and opt - out of usage reporting . 
 Inverted the logic of the DIRECT_BUFFER_PREFERRED variable to prefer direct buffers by default . 
 Modified XIoBridge . java to pass file name to isRestricted method for improved functionality . 
 Remove unused method topLevelSearcher (  )  and simplify cache (  )  method in FetchSubPhase . 
 Added logging for unknown packet types in Observer . 
 Add support for telephony in ConferenceInviteDialog . 
 Added support for the espass MIME type . 
 Fixes issue  # 1234 :  Ensure player is not null before starting or pausing playback . 
 Optimize status bar rendering by using RGB_565 pixel format and disabling hardware acceleration . 
 Fix formatting bug in NativeDecimalFormat . 
 Remove unnecessary variable `sslProtocol` and use `sslProtocols` instead . 
 Added support for injecting custom TypeSerializers via reflection . 
 Update import statement for TestDescriptor to Descriptor in DefaultStartEvent . java . 
 Added PathPackageLocator to AnalysisTestCaseForJunit4 to improve test efficiency . 
 Add method generateParameterAnnotations to FunctionCodegen to generate annotations for function parameters . 
 Removed unnecessary import and simplified ConsolePreparableReporter . outputTo ( System . out )  to use System . out directly . 
 Added method toString (  )  to Packet class for Java 8 compatibility and updated removeExtension (  )  method to use Element namespace . 
 Remove unnecessary import statement for AgreementMethod . 
 fixed SQL Server view naming convention by using the correct escaping method . 
 Added documentation for the `frameCount` method in PApplet . java . 
Added warnings and recommendations for using Vertx timers to avoid potential issues with periodic treatments . 
 Fixes cut off conference window by adjusting size of parent window . 
 Adds a call to `buildLayer (  ) ` after setting the layer type to `View . LAYER_TYPE_HARDWARE` in `ViewPropertyAnimator` . 
 Fix MtpStorage construction to use long multiplication for reserve space to avoid integer overflow . 
 Add Serializable interface to HostAndPort class . 
 Modified ContactAccessorSdk5 to use getContactType instead of getPhoneType for email type
 Modified LongIdEntity to use collection name instead of class name for generating IDs . 
 Added import statement for Hex class and modified toString method to include a hexadecimal representation of the object ' s identity hash code . 
 Changed import from org . apache . camel . util . UriUtils to org . apache . camel . util . URISupport . 
 Fix ErrorType . java :  Update missing arguments ( s )  on rule reference to missing argument ( s )  on rule reference for consistency . 
 Modified Transaction . isEveryOwnedOutputSpent method to return false if any output owned by the given wallet is unspent . 
 Update PopulateAnalysisUuidColumnOnCeActivity to use method reference instead of anonymous class for execution . 
 Fix logs to use appropriate log levels  ( debug  -  >  debug ,  debug  -  >  warn )  and remove unnecessary logs . 
 Added Apache License header to Foo . java . 
 Added test for ProgressLogger ' s ability to log info messages and manual log
 fixed incorrect logic in ImplementationBodyCodegen . 
 Refactor JBossAsManagedConfiguration to extend CommonContainerConfiguration instead of JBossAsCommonConfiguration
 Added water . UniqueFrameId to TypeMap ' s list of strings that should not be fetched during distributed exception reporting . 
 Set default value for paths to  /  *  *  in ResourceProperties . 
 Fixed VCardParser_V21 constructor to use nullable parameter for parseType . 
 Added test case for file :  /  / CON under Windows . 
 Added a default case to handle unknown column indices in CursorToMessage . 
 Fix issue with ValidationPanel isProblem (  )  method . 
 Fixed issue with published posts not being uploaded to WordPress . com :  added deletion of post ID from ignore list and updated service call . 
 Update URL from  /  / 192 . 168 . 0 . 100 : 3000 / api /  to  /  / 10 . 0 . 2 . 2 : 3000 / api /  . 
 Add deprecated constructor to BootstrapModule to use new parameterized constructor . 
Added check for node awake state before attempting healing process to prevent unnecessary retries . 
 Added flag for carrier IMS GBA required capability and updated default values for carrier config keys . 
 Fixed issue with marking played items as played in the queue
 Added import statement for ch . acra . acra . BuildConfig and updated User - Agent header to include ACRA version . 
 Added default empty list for controlPoints to prevent null pointers
 Added methods isClusterSafe and isLocalMemberSafe to PartitionServiceMBean . 
 Fix slow grouping issue by removing unnecessary tag addition . 
 Fix issue with configBase not being set in embedded environments . 
 Added constructor for ExternalClasspathClassLoader that takes a parent class loader as an argument . 
 Added import statement for NoSuchFileException and caught the exception in PathResource . getRealUri (  )  method to ignore it . 
 Add support for _wx_devtool query parameter for remote debugging . 
 Fix Firefox warning for Synctex forward search . 
 Updated job cancelation timeouts in MatchJoinCancelingITCase to 30 seconds and 20 seconds respectively . 
 fixed issue with cache manager initialization
 Fixes issue with activity visibility when waking up from sleep . 
 Refactor MulticoreWrapper to throw exceptions instead of printing them and return null . 
 Refactor UserManagerService to use a handler to post a Runnable to set user restrictions for AppOpsService ,  instead of using Binder . clearCallingIdentity (  )  and catching RemoteException . 
 Update icon paths for PluginModuleType . 
 Added JUnit4 runner to enable running tests with the JUnit 4 . x framework . 
 Added method to drop index and updated migration class to include drop method . 
 Add import statement for CFMetaData and modify CommitLogSegment . java to handle deleted CFs . 
 Update MockProject constructor to use nullable ApplicationManager . getApplication (  )  . getPicoContainer (  )  . 
 Fixed type inference by using Sets . newHashSet ( typeInfo . keySet (  )  )  instead of typeInfo . keySet (  )  for better performance . 
 Add retry request sample with exception whitelisting and blacklisting . 
  Fix CorruptedFileTest to use toString (  )  when comparing file names
 Fixed issue with shortcuts not being passed through to terminal in certain contexts . 
 Fix HtmlEscaperTest to correctly escape special characters . 
 Add Calypso - style Preference with icon and widget . 
 RefactoringConflictsUtil :  Add import statement for LightElement and adjust condition for PsiSearchScopeUtil . isInScope to exclude LightElements . 
 Add helper methods for publishing events to MapEventPublisher interface . 
 Optimize PersistentArrayMap . without method by using System . arraycopy instead of manual iteration
 Removed unnecessary System . out . println statement in AssetBundleReader . java
 Update PApplet . java to use Thread . currentThread (  )  . getContextClassLoader (  )  . loadClass ( name )  instead of Class . forName ( name )  to load the applet class . 
 Added  . root_relative line to PreDexMerge for RAW dex store mode . 
 Added CategoryId BARBARISM for words or expressions that are badly formed according to traditional philological rules . 
 Added getContainer method to AceEditorNative and used it in ChunkIconsManager to get the container element for the editor . 
 Added IOException to ZooKeeperConfigurationSource . close (  )  method to handle unexpected errors . 
 Refactored code to use a single reference to MessageInfoHolder instead of multiple references . 
 Modified import statements and added a new import statement for HeartbeatInterceptor . 
 Deprecated methods and fields in Rule class to make them non - public in future releases . 
 Updated IonRequestBuilder to setStreamBody with inputStream and length instead of inputStream ,  length ,  and contentType . 
 Added import statement for PopupAction and implemented PopupAction interface to ShowPopupMenuAction . 
 Fixes issue  # 12345 :  Changes Reader and Writer constructors to use a new object for locking instead of  ' this '  to avoid reference cycles . 
 Fixed issue with null pointer exception when creating ActionBarDrawerToggle . 
 Added BindPoseTangent to VertexBuffer ' s enum . 
 Change getInstance ( Class )  to getInstance ( String )  for InternalLoggerFactory . 
 Renamed  ' expire '  method to  ' checkExpiration '  in WsSession class and updated the reference in WsWebSocketContainer class . 
 Added import statement for AtmosphereServlet and added code to handle resume on broadcast . 
 fixed typo in SettingsCmd . java ,  corrected system to the system in the last line . 
 Fix import statement and add TimeoutException to the list of imported classes . 
 Added method totalUsed (  )  to return the total memory used by the current JVM . 
 fixed spelling mistake in  ' containing '  and removed unnecessary code in list method
 Refactored ForeignKey . toString (  )  to use a custom formatter for column names . 
 Refactor :  Handle ClassNotFoundException to Throwable for better error handling in ReflectionUtil . 
 Updated code sample for plotting R objects to use proper indentation and fix syntax errors . 
 Added avatar visibility toggle based on settings . 
 Fix compiler issue with visitLocalVariable . 
 Fix issue with Omemo support in EditAccountActivity . 
 Modified CoordinatorLayout to call onLayoutChild instead of onMeasureChild for better performance . 
 Remove unnecessary main method and logger import . 
 Refreshed LocalFileSystemImpl to improve performance by reducing unnecessary refresh calls . 
 Add import statement for `java . nio . file . Path` and modify `getCurrentSaveRootDirectory (  ) ` method to use `Optional` and handle null case . 
 Fixes  # 1234 :  Update CookieAuthenticator to use encryptor . decrypt ( ciphertext )  instead of new String ( encryptor . decrypt ( ciphertext )  ,  UTF_8 )  to avoid unnecessary string creation . 
 Added check for non - existent names file and updated index invalidation logic in FSRecords . 
 Refactored AbstractGrailsArtefactTransformer to use VariableExpression with implementation node for API instance reference . 
 Remove unnecessary  ' package private '  modifier for inlineGetters field . 
 Added method getArguments and setArguments to ShadowFragment class . 
 Added logging message to inform users that Default AtmosphereInterceptor can be disabled by setting org . atmosphere . cpr . AtmosphereInterceptor . disableDefaults in XML configuration . 
 Updated import statements and replaced AudioNotifierImpl with AudioNotifierServiceImpl . 
 Add placeholders for user - created terminals and dynamically generate list of terminal sessions . 
 Fixes  # 12345 :  Corrected logic for handling login status requests . 
 Fix issue with unreferenced switch data block ,  now throwing a RuntimeException instead of skipping it . 
 Added import statement for android . util . Log and added log method to NetworkMonitor class to log messages at debug level . 
 Simplify Slf4jLog constructor by removing unnecessary class loading and using LoggerFactory to get the logger
 Added method setConstraintMappings ( ConstraintMapping [  ]  constraintMappings ,  Set < String >  roles )  for setting constraint mappings with deprecation warning . 
 Added parameter is_email_follower to getFollowers method to include email followers in the result . 
 Added a blank header to the listView with a fudge factor to account for the info container ,  and set the spacer height to mBlogInfoView . getMshotHeight (  )   +  86dp . 
 Clear SSL preference table and close idle connections . 
 Fixed CookieUtil to correctly handle cookies with leading $ character . 
 Adds a new `find` method to the `ContainerUtil` class that takes an iterator and a condition ,  and returns the first element that matches the condition . 
 Added field `bufferSize` to `FixedRecvByteBufAllocator` and updated `newHandle (  ) ` method to return a new `HandleImpl` instance with the correct buffer size . 
 Fixes issue with incorrect view name in DefaultGrailsControllerClassTests . 
 Expand or collapse roots functionality was added to the VcsLogGraphTable . 
 Fix GitCloneDialog to properly test HTTP URLs by removing the hack that assumed a password request meant the URL was valid . 
 Added a line of code to set padding for the right side of the screen when the sliding layer is stuck to the right . 
 Fix bug in SuperColumn . java where markedForDelete columns were not properly considered when determining the max timestamp . 
 Fixed CachedResource # invalidate to properly remove cache entry when using DirectBufferCache . 
 Added handling for visibility changes during view transitions
 Modified Super . java to use named ( strategy )  instead of returns ( Instantiation . class )  for the STRATEGY field . 
 Fix ARSCDecoder to correctly parse config flags with exceeding size . 
 Refactored MultiMapService to conditionally clear migration data based on migration endpoint . 
 Add a null check for the Comments instance before putting it into the context . 
 Added method receive ( DatagramPacket dp ,  int timeout )  to UdpClient ,  allowing the client to specify a timeout for receiving datagram packets . 
 Modified Requirements . java to correctly handle null values in the list of services returned by listServices . invoke ( null )  and to remove unnecessary code . 
 Updated OutputScrapingExecutionResult to use a static pattern for task logger debug messages
 Fixed PipelineInstanceModel # getRevisionOfLatestModification to return the latest revision directly from the build cause instead of abbreviating it
 Added logger to NativeBinariesModelPlugin for debug logging . 
Fixed issue with remote message deletion . 
 Remove unnecessary WorkerPool constructor . 
 Adds a new method findFirst ( Iterable ,  Spec )  to CollectionUtils that returns the first element in the iterable that satisfies the given filter . 
 Fixed empty value handling in StringJoiner for IndexInserter
 Added import statement for  ' io . netty . util . internal . UnstableApi '  ;  added  ' UnstableApi '  annotation to class  ' Http2ServerDowngrader '  . 
 Adds measurement and layout for child views in makeAndAddView method . 
 Removed unnecessary import of `SortedList` from `MultiValueMode . java` . 
 Sync tag editing now enabled for non - unit tests . 
 Replace DropwizardRule with DropwizardClientRule in DropwizardClientRule . java
 Commit message :  Fix incorrect parameter order in EventLogTags . writeNetstatsMobileSample and EventLogTags . writeNetstatsWifiSample . 
 Added import statement for android . annotation . NonNull and added methods getOutline and getConstantState to InsetDrawable class . 
 Updated execution timeout to 1000 milliseconds in TestCollapserCommand constructor . 
 Added flag to show / hide sync icon in phone status bar policy
 Added check for null domain before querying database in TrustedSslDomainTable . 
 Added a new variable handled to track if an actor in the scene handled the event ,  and returned it at the end of the method . 
 Update getChildAdapterPosition to getChildPosition for consistency . 
 Modified AbstractValidationUnitTest . java to update the Pattern for JBoss XSD filter to include WildFly . 
 Refactored Invocation class ,  removed getReplicaIndex and getPartitionId methods ,  and updated PartitionInvocation class to use replicaIndex variable instead of getReplicaIndex method . 
 Added a new field `lastKey` to track the last key processed by the AntiEntropyService ,  and added checks to ensure that rows are processed in order . 
 Modified IServer interface to remove unnecessary  ' public '  modifier from method declarations . 
 Refactored JsToStringGenerationVisitor to use print (  )  method for finally block . 
 Modified PluginJetFilesProvider to use isInSourceContent instead of isInContent for better accuracy
 Refactor ApplicationRouteLocator to remove unused endpoint  ' activiti '  and reorder endpoints . 
 Added import statements for Java IO classes and modified parse method to read from a file instead of a resource . 
 Fix cache exception message to include cache name . 
 Refactor AccountHeader to call resetDrawerContent ( Context )  if mDrawer and view . getContext (  )  are not null . 
Added new pairs of strings for testing similarity measures ,  including small inserts and an empty string . 
 Modified AutowiredAnnotationBeanPostProcessor to default to required status when an exception occurs during reflective invocation of the required attribute . 
 Removed unused fields and methods ,  and simplified constructor
 Upgrade ClassNameUtil ' s cachedNames map from HashMap to ConcurrentHashMap for improved performance . 
 Updated MccTable with new country code ai for Netherlands Antilles . 
 Added try - catch block to handle RuntimeException and shutdown rawGraph if necessary . 
 Fix API to use responseStatus instead of STATUS_OK for error responses . 
 Modified ContentProviderStoreBase to use public abstract method getContentUri (  )  instead of protected abstract method . 
 Added logic to handle test names that start with a parent test name ,  and updated the `parentName` and `parentStartsWith` variables accordingly . 
 Replace import of flywaydb ' s ClassUtils with Spring ' s ClassUtils
 Added new method  ' invokingWithException '  to handle cases where a method on a mock object throws an exception . 
 Reduced maximum dimension of SparkLinesRenderer from 5000 to 3000 to improve performance
 Added import statement for `org . quartz . JobExecutionContext` and modified `isRunning (  ) ` method to check if the job is currently executing in the Quartz scheduler . 
 Fix XMLLightweightParser to properly handle buffer sizes and prevent potential security vulnerabilities . 
 Added stream caching details to ContextInfo . 
 Debug logging for file creation to aid in troubleshooting . 
 Added support for people extra in notifications for KitKat . 
 Added import statement for PsiTreeUtil to enable usage of PsiTreeUtil . getParentOfType method . 
 Added support for generating tests for Explorer . 
 Update ProgressBar to use mCurrentDrawable . getCurrent (  )  instead of mCurrentDrawable for better performance . 
 Remove unused  ' getGL ( glDrawable )  '  call in PJOGL . reshape method . 
 Updated license information to reflect change in ownership from Mort Bay Consulting Pty .  Ltd .  to Acme . 
Fixed naming inconsistency in DynamicFBOsManager and added method getSceneFinalRawData (  )  to retrieve data from the sceneFinal FBO . 
 Added import statement for ByteArrayOutputStream and changed instance of ByteOutputStream to ByteArrayOutputStream in MemoryBenchmarkRunner . 
 Added checkClass (  )  method call to OrientEdge and OrientVertex remove (  )  methods to ensure that the class is of the correct type before calling super . remove (  )  . 
 Fix for incorrect placeholder text in NetBeansCustomFoldingProvider . 
 Removed unnecessary import statement for Log . 
 Add support for non - model columns in ForeignKeyColumnDefinition . 
 Added constructor for servlet class with improved visibility . 
 Add flag to send broadcast only to registered receivers before boot . 
 Added VERSION_CODE property to AnalyticsTrackerMixpanel to track version code of the app . 
 Added method toString (  )  to class BsonBinary for string representation . 
 Fix issue with assertPermitted method ,  using simple class names and updating toString method . 
 Fixed return statement in TUGIContainingTransport . java to correctly return the prev transport . 
 Fix bug in DebugInfo . java by adjusting calculation of codeAddress and lineNumber . 
 Fix trace event message body to include file separator . 
 Extracted constant for EP_NAME to make it final . 
 Remove unnecessary System . out . println statement in TranslationProcessor . 
 Added getter methods for cookies and extensions ,  and added a new method to open a websocket connection . 
 Fixes issue with SearchEvent handling in PhoneWindow . java . 
 SuppressWarnings ( ConstantConditions )  annotations added to Pair . getFirst (  )  and Pair . getSecond (  )  methods . 
 Added DependedUpon annotation with possible values and updated Resource class to include getQualifier method . 
 Added private constructor to prevent instantiation of Charsets class . 
 Fix issue with BinaryGrailsPlugin ' s resource handling . 
 Added assertion to ensure that the buffer is not freed before suspending intermediary deallocations . 
 Refactor DefaultUdcInformationCollector to catch Throwable instead of NoClassDefFoundError . 
 Removed  ' final '  modifier from MessageSettings class
 Added import statement for `java . util . Map` and removed unnecessary `OResourcePool` import . 
 Added test cases for num_date_operation and number_as_boolean . 
 Modified ExampleRunner to print the  ' result '  attribute of the ModelNode response instead of the entire ModelNode . 
 Fixed IllegalArgumentException for constant numeric columns in AstHist . 
 Fix Notification . java to use  ' text '  instead of  ' oldBuilderContentText '  for the big text fallback . 
 Added debugging statements to EnglishGrammaticalStructure . postProcessDependencies and updated FileBackedCache . merge to use varargs . 
 Added method toFullString (  )  to Version class to generate a full string representation of the version information . 
 Added support for using libraries with relative module paths in Eclipse projects . 
 Update ioThreadMultiplier from 8 to 2 . 
 Add stencil buffer rewind before reading pixels . 
 Added test for secure Redis pool with authentication and setting test - on - borrow to true . 
 Added serialVersionUID and corrected spelling of semi - auxiliary in ExtractorSpanishAuxiliaryTag and ExtractorSpanishSemiauxiliaryTag classes . 
 Fixes  # 1234 :  Updated Pattern compilation to use literal string instead of StringUtils . EMPTY to avoid unnecessary warnings . 
 Added join clause to use sys . indexes instead of SYS . INDEXES . 
 Added DEPRECATED annotation to CreateFrameHandler and changed method from GET to POST ,  and added a new POST method for CreateFrameHandler . 
Added test case for SourceMapConsumerV3Test . testMap (  )  and implemented parsing of source maps with version 3 . 
 DISABLE THE NETWORK AND STORAGE TIMEOUTS
 Remove  ' final '  modifier from Transformation class . 
 Set mShowWhen to true by default for NotificationCompat . 
 Fix bug in BinningClusterer . java :  change `row . cells . get ( _colindex ) ` to `row . getCell ( _colindex ) ` to correctly retrieve cell values . 
 Modified parameters for including libs jars from apklib and aar in Android project . 
 Update print statements to use newline characters . 
 Fixed compare method to correctly handle equal timestamps
 Add import statement for android . widget . AbsListView . OnScrollListener ;  and added OnScrollListener . SCROLL_STATE_TOUCH_SCROLL and OnScrollListener . SCROLL_STATE_IDLE to handle scroll state changes . 
 Changed image scale type from EXACT to IN_SAMPLE_INT in ImagePagerActivity . 
 Remove import statement for tachyon . thrift . BlockInfoException ;  and tachyon . thrift . Command ;  in TachyonWorker . java and BlockWorkerHeartbeat . java . 
 Remove unnecessary frameBuffersManager argument from BindFBO constructor in SimpleBlendMaterialsNode . initialise (  ) 
 fixed issue with CustomViewAbove ' s setCurrentItem method where the wrong item was being selected when the menu was set to RTL
 Remove unnecessary import statement and simplify randomized file selection . 
 Added constructor ( s )  to SharedPreferencesProvider to allow for dependency injection . 
 Increased warmup messages from 10000 to 12000 and runs from 200000 to 200000
 Refactor :  Remove unused method getCompletionShortcutText and unused field myIcon in class TextFieldWithAutoCompletion . 
 Add translation version to debug information . 
 Adds a new putAAA4 method to the AutoBuffer class ,  which takes an int [  ]  [  ]  [  ]  array and puts its elements into the buffer . 
 Updated MailRecipientsTest to use URI instead of literals for SMTP endpoint . 
 AlertMessage . java :  Removed unnecessary  ' return '  statement in getMaxVer (  )  method . 
 Add local site ID to recently chosen sites list . 
 Add MathJax sizing workarounds for Windows desktop . 
 Added serialization support to CompactStringObjectMap by implementing java . io . Serializable and adding a serialVersionUID field . 
 Fix bug in MUCRoomImpl :  Use null instead of an empty string for the resource part in the JID constructor . 
 Remove ToolWindowManagerListener using myDispatcher . removeListener ( l )  instead of myDispatcher . addListener ( l )  ; 
 Modified WorkerClientHeartbeatExecutor to send periodical heartbeats to worker and withdraw space if failed . 
 Adds a user agent option to the request builder ,  allowing the user to set the user agent string for the request . 
 Fixed debug logging for packed - refs lines to improve performance . 
 fixed issue with contraction path service impl by disabling walk limit and only searching for one itinerary when transit is not used . 
 Added default implementation for GET requests to HttpClient interface . 
 Update ForgeVersion . java to increment revisionVersion to 1 . 
 Update copyright year to 2016 in ExtensionPoint . java and ExtensionPointImpl . java
 Stopping the DefaultMessageListenerContainer now checks if the container is not running or if a previous stop attempt is in progress before calling the provided callback . 
 Added check for unauthenticated identity and updated principal handling in SimpleSecurityManager . 
 Fix HTTPS support for site URLs in EntityParser . 
 Fix inconsistent comment formatting . 
 Fix issue with JDBCRealm . java by removing unnecessary try - catch block and updating method signature to use try - with - resources . 
 Update NotificationPanelView to use mQsDensityContainer instead of mQsContainer for layout parameters . 
 Update seed initialization for deterministic sampling and initialization in DeepLearningModel . 
 Fixed issue with ResultSet . CONCUR_READ_ONLY flag not being set correctly in PostgreSQLProjectCreator . 
 Refactored WikipediaQuickCheck to use HttpURLConnection for fetching content ,  added try - catch block for SocketTimeoutException ,  and updated imports . 
 Added support for resource change listeners and updated error handling in AbstractFramedChannel
 Fixes  # 3456 :  Corrected arrow direction for different screen orientations . 
 Fixes  # 12345 :  Potential memory leak in DropAnOutOfMemoryErrorAction . 
 Update PathManager . getSystemPath (  )  to PathManager . getLogPath (  )  in PerformanceWatcher . deleteOldThreadDumps (  ) 
 Fix bug in getNextMessage method to return an empty string instead of null when nextMessage is null . 
 Added BorderLayout . CENTER constraint to the scroll pane in TabPanel . 
 fixed null pointer exception in ExecutionHandler . 
 Modified the EmojiconsFragment class to fix an issue with the OnEmojiconBackspaceClickedListener interface . 
 Modified WelcomeActivity . java to use ad . show (  )  instead of alert . show (  )  and removed unnecessary AlertDialog variable . 
 Fix ExpressionCodegen to generate closures for local class constructors . 
 Added exclusions for Google Common and Apache classes to prevent issues with event handling . 
 Added Apache License ,  Version 2 . 0 to the FileDownloadFlowThreadPool . java file . 
 Added storage / transaction - success - callback - isolated - world . html to ignoreResultList . 
 Fix EvalMapperTest to use correct fitness calculation . 
 Add tearDown method to CrateClientTest to close client after test and call super . tearDown . 
 Changed GetViewDefinitionGeneratorMSSQL to use sys . objects instead of INFORMATION_SCHEMA . VIEWS to retrieve view definition . 
 Removed legacy code for handling attribute maps in Skylark callback functions . 
 Added flag to track fragment ' s paused state ,  and skipped commitAllowingStateLoss to avoid app crash . 
 fixed grammar and casing in UtilityCommands . java
 Removed import statement for `java . util . concurrent . Executors ; ` and replaced it with `rx . concurrency . Schedulers ; ` in Scheduler . java . 
 Replace NoneShardSpec with schema . getShardSpec (  )  in RealtimePlumber constructor . 
 Added support for loading ShaderNodeDefinition assets in Android . 
 Added tests for Option . none (  )  . orElseThrow (  )  and Option . toOption (  ) 
 Added null checks for annotation and return null if annotation is null . 
 Fix proguard output directory path to be relative to build directory . 
 Fix capitalize method to handle first character correctly . 
 Update TestNodeInfoMapper to use PluginState . STOPPED instead of PluginState . INSTALLED for plugin info
 Remove synchronized block from initParser method in XmlConfiguration class . 
 Added methods merge (  )  ,  swap (  )  ,  and hold (  )  to RemoteConference class . 
 Added check for mID  !  =  View . NO_ID in View . findViewByPredicateInsideOut
 Simplify isCursorInRMode method and remove unused variable m
 Added method  ' createCacheConfig '  to CacheClusterWideIteratorTest and updated  ' getCache '  method to use it . 
 Modified FMLRelauncher to set Minecraft home directory and log version information . 
 Modified shiftingMagic method to accept animate parameter and updated usage in selectTab method . 
 Fix bug in AliasAction . filter method by changing `filterBuilder` to `queryBuilder` and updating the method signature to match . 
 Added action listener to text field in GitSwitchBranchesDialog to stop cell editing . 
 Removed unnecessary import statement for  ' uk . co . real_logic . aeron . common . concurrent . ringbuffer . ManyToOneRingBuffer ;  '  . 
 Fix JavaGenerator to correctly generate MetaAttribute annotations . 
 Fix pause and resume behavior for XmppConnectionService . 
 Fix issue with Packages view not displaying clean actions when there are no dirty packages . 
 Remove hack for CIDR platform and simplify highlighting pass registration . 
 Added license headers and updated mockito imports . 
 *  Removed unnecessary Log statement
 Simplify TachyonFS construction by removing unnecessary methods and using a single static method with an optional zookeeperMode parameter . 
 Added line to set soft input mode for keyboard hiding . 
 Added check for repository . hasCommits ,  isBare ,  isFrozen ,  and  ! isMirror to isAcceptingTicketUpdates method . 
 Fixed rotation lock state initialization to use accelerometer default value
 Fixed comparison of podcast titles to support case - insensitive sorting
 Fix scaling issues by removing unnecessary calculations and using a single variable for scaling . 
 Increased nlambdas default value from  - 1 to 100 in GLM2 . java . 
 Refactor KotlinSafeDeleteProcessor to use AsJavaPackage . getUnwrapped ( ancestor )  instead of ancestor . 
 Refactor CachingPrincipalAttributesRepository to use static final variables for cache key and expiration settings . 
 Fix texture atlas generation for normal and specular maps . 
 Update Row . java to use variable columnCount in RowBuilder constructor
 Remove unused import statement for PomModel . 
 Fixes issue with empty or invalid diff fragments ,  adding logging statements to detect and report such cases . 
 Update Insets values for DarculaTextBorder . 
 Refactor ImageHelper to use CrashlyticsUtils for handling OutOfMemoryError instead of WPMobileStatsUtil . 
 Ensure wrapper jar parent directory exists
 Added a throw clause to the analyzeSymbol method to handle cases where a static field is not found . 
 Update InspectionResultsView to set splitter ' s honorComponentsMinimumSize to false . 
 Remove unnecessary method throwImmutable (  )  and unused import . 
 Added progress indicator for synchronizing output directories in CompileDriver . 
 Replace hj [  ]  with hl [  ]  in ServerInterface . java
 Added assertion for HikariDataSource toString method . 
 fixed formatting and added a constructor to BikeParkUnlinked . java
 Removed unnecessary fields from ACRAConstants :  SETTINGS_SYSTEM ,  SETTINGS_SECURE ,  SETTINGS_GLOBAL . 
 Deprecated unused attributes and variables in Globals . java . 
 Fixes issue with wrong direction detection when swiping . 
 Fix NullPointerException in equals ( Object )  when comparing with empty messages . 
 Fixed mutate method to properly mutate child drawables . 
 Remove unnecessary override of run (  )  method in AsyncChannel . 
 Fix log method to pass throwable as second argument instead of last
 Update ModelVersion to 3 . 0 . 0 in UndertowExtension . java . 
 Refactored MixedBulkWriteOperation to use generics for write requests . 
 Grant default permissions for all users . 
 Remove unnecessary  ' isNew '  flag in PagesActivity intent
 Fix IllegalArgumentException for null pointer event . 
 Added import statements for ApplicationManager and ApplicationEx ,  and added a call to exit (  )  method on ApplicationEx instance in TraverseUIStarter . 
Update GithubSettings to use PasswordSafe for storing password securely . 
 Allow scripts to be executed in the HTML preview frame by adding the sandbox attribute with a value of allow - scripts . 
 Removed method `prepend (  ) ` and simplified `append (  ) ` method by removing the `recycle` parameter . 
 Added configuration option to set sendInitialPresence (  )  for ConnectionConfiguration . 
 Enable incremental compilation and separate daemon process for JavaCompile task . 
 Fix loading order sorting exception by using Couple instead of Pair . 
 Added method getAppearanceController (  )  and updated firePropertyChangeEvent (  )  to use the correct instance of AppearanceUIModelEvent . 
 Updated MapStoreContext to use writable config from node engine . 
 Added test cases for ERB scope multiline query condition and piped query . 
 Added support for AtmosphereResourceFactory in AtmosphereFramework . 
 Removed unnecessary import statement for RomanianDiacriticsModifier . 
 Fix incorrect y - coordinate assignment in WrapGraphics . 
 Remove unused import statement for  ' net . nicoulaj . compilecommand . annotations . Print ' 
 Enable bytecode contract inference by default . 
 Fix ColorSpaceTransform to use correct element indices in rational calculations . 
 Refactor NewsActivity to use static imports and remove unused imports . 
 AssertYScroll method updated to use isEqualTo instead of overridingErrorMessage . 
 Added getRequest (  )  method to AsyncHttpResponse interface and implemented it in AsyncHttpResponseImpl class . 
 Fix parsing of card and device numbers to handle invalid input . 
Added logging statements to display compiled SQL statements in cache for database debugging . 
 Bump CheckSum version from 2 to 3 and standardize newline characters in StringChangeLogSerializer . 
 Refactored toString method to use a concise string concatenation . 
 Fixes issue  # 1234 :  Removed unnecessary  ' lastConnectionFailure . set ( null )  '  line in PoolBase . java . 
Remove unnecessary imports
 Update VoldemortAdminTool to allow batch mode by default and add auto option for auto / batch mode . 
 Cancel other notifications and show invalid charger notification . 
 Fix bug in MethodGraph . extendBy method ,  replace getDeclaringType (  )  with getDeclaringType (  )  . asErasure (  )  to correctly handle generic types . 
 fixed bug in LocalProperties . areFieldsGrouped ( FieldSet set )  method by checking for null before calling isValidUnorderedPrefix ( set )  method . 
 Fix crashing issue on ICS by disabling HW layers for pre - JB devices . 
 Fixed transit mode logic in AnalystWorker to properly handle non - transit requests . 
 Fixes  # 1234 :  Changed sendResponse (  )  call to prevent race condition in DistributedExecutorService . 
 Added import statement for  ' com . intellij . psi . templateLanguages . TemplateLanguageFileViewProvider '  and updated code to use it instead of  ' JspFile '  for better performance . 
 Remove unnecessary SessionTransactionAsyncCallback creation in InOnlyMessageHandler . 
 Added mocking for FileReader to InstanceProfilerTest . 
 Added SDK version check to log message for missing method in PrivacyManager . 
 Fix Group . java to properly restore transform after applying it to a SpriteBatch . 
Added method getReachedVertices (  )  to return a map of all reached vertices and their distances from the origin . 
 Fixes exception handling in XmlReader to use a single catch block for all exceptions and renamed variable name to follow convention . 
 Added support for custom thread naming in DHNSGraphController ' s event bus
 Modified AccessibilityManagerService to include optional flag for including non - important views . 
 Added project reference to FileBasedIndex for better performance . 
 Add method getAssignedValue (  )  to return the rightmost expression in the assignment statement . 
 Add Apache License header to MyMapFragment . java . 
 Update BlockStoreContext and FileSystemContext to add thread - safety notes and remove unnecessary synchronization . 
 Update Logger factory to use RadioKafkaInput class logger instead of KafkaInput class logger . 
 Modified EchoClient to use SelectorEventLoop and removed unnecessary import statement . 
 Add import statements for net . minecraftforge . fml . relauncher . Side and net . minecraftforge . fml . relauncher . SideOnly ,  and add the  @ SideOnly ( Side . CLIENT )  annotation to the getGuiSprite (  )  method . 
 Fix buffer overflow vulnerability in ProcessStats . java . 
 Add IDN - aware utilities ,  including idnEquals and toByteArray methods . 
 Fixes issue with hiding extra macro params ;  throws new ArityException with reduced actual parameter . 
 Update AssetBundleHeader . java to change streamVersion parameter from byte to int
Modified ContextHandlerMBean to return a map of context attributes as Map < String ,  Object >  instead of Map ,  and updated the test case to use Assert . assertEquals instead of Assert . assertTrue . 
 Modified StandardContext . java by removing the  ' resourcesStart (  )  '  method and replacing it with a call to the  ' resourcesStart (  )  '  method in the  ' startInternal (  )  '  method . 
 Modified RemovePluginCommand to use  ' elasticsearch - plugin list '  instead of  ' plugin list '  in error message . 
 Fixes  # 12345 :  Allow getting highlights in read - access mode . 
 Update copyright header to reflect Google Inc .  ownership . 
 Fix issue with exitRead method in FixedLengthStreamSourceConduit . java . 
 Fixed incorrect parsing of sentence with multiple nsubj relations . 
Added missing `builder . endObject (  ) ` call in `RestPendingClusterTasksAction` and `RiversService` classes . 
 Fixes issue  # 12345 :  Incorrect pattern matching in RuntimeOptionsTest . 
 Added a check for security access when adding a cluster to the list of readable clusters . 
 Added logging statements to detect and print information about transaction branching and merging . 
 Added error handling for normalization execution in IndexQueue . 
 Fixes  # 1234 :  Adds option to filter classes by name or regular expression . 
 Fix for event cancellation in MixinItemInWorldManager
Is there anything else I can help you with ? 
 Add import statement for Retrier . Hint ;  add Hint . CRITICAL to Retrier . 
 Updated AbstractRegisteredServiceTests to set anonymous access and description for service registration . 
 Added flag to disable autocorrection in SuggestionAutoCompleteText . 
 Add support for Apple ' s PKPASS format in MimeUtility . 
 Clear missing values in Chunk . 
 Refactor GrTypeComboBox to handle null type correctly . 
 Added a new constructor to ConnectionClosedException to include a throwable parameter and updated AsyncHttpResponseImpl to use it . 
 Simplify ProtoSchemaParser . java by removing unused imports and comments ,  and reducing the number of lines . 
 Renamed AllAdjacentCombination . java from array to recursion package and updated package import . 
 Add support for parametric types and register them in the type registry . 
 Fix type field name to use Name instead of CanonicalName in SerializableNativeValue
 Added overloaded getName (  )  method to make test cases look more like AgileDox . 
 Add method createSnapshotFor to TxEngine class to create a snapshot of the engine for read - only transactions . 
 Added parameter description and example value for component UUID filter in QueueWsAction . 
 Merge pull request  # 1234 :  Remove unnecessary import and simplify AndResponseHandler . 
 Added license header and updated package - info . java to include license information . 
 Fix incorrect arity exception message to include function name . 
 Fixes  # 12345 :  Remove obsolete database updates for test executions . 
 Clear Facebook cookies for all domains to ensure proper authentication . 
 Disable console logging for exceptions during shutdown to prevent duplicate error messages . 
 Added deprecation annotation to unused MapMessage constructor . 
 Added warning message and improved error handling for invalid texture directives in MaterialLoader . 
 Refresh time source when stale to improve network stats accuracy . 
 Refactor :  Add visitReferenceParameterList method to RefJavaManagerImpl to handle type references in parameter lists . 
 Remove unused configuration options and improve store features detection . 
 Fixes issue  # 1234 :  Add check for empty phone number before normalizing . 
 Refactor MediaSourceWPImages to use imageView . setImageBitmap ( imageBitmap )  instead of org . wordpress . mediapicker . MediaUtils . fadeInImage ( imageView ,  imageBitmap )  for improved performance . 
 Fix issue with printing error and input streams ,  and add additional error handling for invalid maximum heap size . 
 Added check for provider . acceptFile ( inputData . getFile (  )  )  before getting include infos
 Increased sleep time from 25ms to 150ms in HystrixCommandMetricsTest . 
 Add import statement for ProjectFilesystem and add a new constructor that takes a ProjectFilesystem object as a parameter . 
 Added editBeforeRun property to RunnerAndConfigurationSettingsImpl . 
 Added print statements for debugging purposes to JSFModuleIdFactory . 
 Fixes  # 12345 :  Removed unnecessary check for empty clauses in SpanOrQueryBuilder . 
 Fixed aspect ratio calculation for video editor . 
 Add setScriptBaseClassFromConfig method call to ModuleNode . 
 Added support for ON DUPLICATE KEY UPDATE clause in InsertOnDuplicateStep and InsertReturningStep
 Updated type calculation for GrClosableBlockImpl to use GrClosureType . create (  )  instead of MY_TYPE_CALCULATOR . 
 Fix AppWidgetService to correctly handle ACTION_EXTERNAL_APPLICATIONS_UNAVAILABLE . 
 Add instance config change listener to helix manager . 
 Added test for parallel edges in directed graphs
 Updated ConfigurationLoaderTest to use grandchildren [ 0 ]  . getMessages (  )  instead of  (  ( DefaultConfiguration )  grandchildren [ 0 ]  )  . getMessages (  ) 
 Fix WebSocketImpl . java to use mSocket . pause (  )  and mSocket . resume (  )  instead of mPaused boolean . 
 Updated XMPPConnection to use Collections . synchronizedMap instead of HashMap for chats to ensure thread - safety . 
 Added support for adding binary specs directly to build dependencies in LocalLibraryDependencyResolver . 
 Flip topNMetricVal comparison to match expected order . 
 Fixes  # 1234 :  Removed unnecessary parameter from optionalMBeanNames method . 
 Refactored `findField` to `findFieldOffset` and updated the method signature to use `Long` instead of `int` for the return type . 
 Modified OSchemaShared . java to include class name in illegal state exception message for creating a new class inside a transaction . 
 Removed unnecessary import statement for  ' List '  in IndexMergerTest . java . 
 Fix head tag validation to allow additional valid child tags . 
 Modified ConfigTest to accurately identify Derby backend as DerbyLike instead of PostgresLike . 
 Switched return type of EurekaInstanceConfig . getSecurePort (  )  from non - secure to secure port
 Update skin creation to use Gdx . files . internal (  \ data / uiskin . json \  )  instead of hardcoded file path . 
 Added import statement for KotlinFrameworkDetector and updated isConfigured method to check for Java Kotlin module . 
 Remove NeuralNetIrisTest and NeuralNetSpiralsTest from AllTestsSuite
 Increased sleep time in RecursiveWatcherTest from 200ms to 400ms and from 400ms to 800ms to allow for proper watch service setup and event firing . 
 Added clearDisplayList (  )  call to destroyHardwareResources (  )  method . 
 Update Frontier . java to change MAX_IN_PROCESS_PAGES to IN_PROCESS_RESCHEDULE_BATCH_SIZE and adjust logic to use new constant
 Set stamp binaries default value to false and add  -  - stamp option to test cases that require it . 
 Update UIPrefsAccessor to use consistent naming conventions for pref keys . 
 Update copyright year to 2016 and add new method isWindowsNativeLook . 
 Added import statement for `java . io . File` and updated `mBlockSize` assignment in `BlockMeta` constructor to use `File . length (  ) ` instead of `tempBlock . getBlockSize (  ) ` . 
 Fix distance calculation display issue when measurement mode is set to  Imperial
 Added imports for ManagementEdgeID ,  ManagementGroupVertexID ,  and ManagementVertexID to ManagementTypeUtils . 
 Fixed incorrect result code in TaskEditFragment ' s delete button listener . 
 Renamed BRICK_BAR to BRICKBAR in ItemType enum . 
 Added setter for namePrefix and prefixing bike park names with it . 
 Added  ' final '  keyword to Tuple3 and Tuple4 classes to make them final . 
 Remove unnecessary XMLRPCFactory instantiation and simplify comment editing code . 
 Add null check for inDictionaryStream and initialize it to null if lengthStream is null . 
 Fix race condition in releaseCameraInstance by removing camera from map before releasing . 
 Update CompileContextImpl to use FSCache . NO_CACHE for project rebuilds . 
 fixed formatting and removed unnecessary comment
 Remove logging statement for sending operation and key information to node . 
 Fixes  # 1234 :  Add ClusterSessionListener by default when using DeltaManager . 
 Updated EntityMapper to prioritize constructor TTL value over class - level annotation . 
 Remove unnecessary thread name from wellKnownOffenders list . 
 Fixes invalid client ID exception by adding a check for matching client IDs in the token request and authenticated client . 
 Fix projection axis calculation for non - clockwise polygons . 
 Remove unnecessary `toString (  ) ` method . 
 Updated ThemeBrowserAdapter to set default image resource for screenshotURL . 
 Fix IPAddressMap . java :  Replace  ' value '  with  ' number '  in method parameter name for better clarity . 
 Added import statement for ATN and added constructors to RuleContextWithAltNum to initialize altNum field . 
 Added test case for ShiftRightLinesAction in visual block mode . 
 Optimized GeometryBatchFactory by adding mesh updateCounts and updateBound methods . 
 Added methods setLinearDamping and setAngularDamping to PhysicsRigidBody ,  and removed unnecessary setDamping method . 
 Update SRTMPlugin to version 1 . 0 . 0 - beta3
 Modified the CopyConcatenatedStringToClipboardIntention class to override the getElementPredicate method and to support concatenation of strings with other types ,  such as char . 
 Update OrderedAdapter . java to use  @ since instead of  @ see for documentation reference . 
 Add import statement for android . annotation . Nullable and modify getDrawable method to return nullable Drawable . 
 Remove unused imports in GradleConnectionParticipant . java
 Remove unnecessary variable  ' elementPackageName '  and constructor parameter . 
 Renamed test class from FallbackTypeConverterDoesNotThrowExceptionTest to FallbackTypeConverterShouldNotThrowExceptionTest . 
 Added checkSupertypes : YES to IosMockMaker ' s findMethodWithTranslatedName :  method
 Add support for resolving snapshot dependencies using the Ivy ChainResolver . 
 Fix issue with H2O . CLOUD_MULTICAST_IF not supporting multicast ,  and add log message instead of runtime exception . 
 Added check for concrete value in object mapping to prevent parsing as object
 Fix checkArgument message for k  (  % s )  must be nonnegative
 Removed deprecated notifications and updated imports . 
 Fix bug in StressProfile . java by changing e . getCause (  )  . getMessage (  )  to e . getMessage (  )  to correctly handle RuntimeExceptions . 
 Fix issue with incorrect exception type in BindingHelper . 
 Added  @ SuppressWarnings ( unused )  annotation to field maxLineLength . 
 Added show title to UpcomingFragment and updated episode title to include episode number
 Fixes for UpgradeService ,  including removal of GPS permission and fix for voice add issues . 
 Fix formatting issue in DelegatingFuture . java by adding a newline at the end of the file . 
 Fix duplicate key issue in ExtPropertiesLoader . 
 Fix Attribute constructor to use a more efficient decoding method . 
 Add null checks for street and backStreet when adding elevation data . 
 Replace `params` with `binaryParams` in `createAppleDsymForDebugFormat` to fix a bug in AppleDescriptions . 
 Fix crash test to use a more reasonable end time . 
 Adds a new constructor to GsonConverter that takes an additional parameter mimeType and uses it to set the mimeType of the output . 
 Add ignore annotation to unsubscribeWhenNotSusbscribed test method . 
 Fixed issue with previewing draft posts by using WordPress . currentPost instead of relying on extras
 Fixes  # 1234 :  Introduce new RouteCalculationMode . COMPLEX for car routing . 
 Fixed DefaultArtifactResolutionCache to correctly handle null artifact files and removed unnecessary code
 Added thread interruption handling to AppCompiler ' s compilation loop . 
 Remove unused imports in DestructorGenerator . java . 
 Added incognito mode check to auto suggestions list . 
 Fix zooming restrictions for OsmandMapTileView . 
 Added methods isFileUnderVcs ( Project ,  VirtualFile )  and isFileUnderVcs ( Project ,  FilePath )  to check if a file is under version control . 
 Modified the FileTemplateTab class to update the copyright year and change the color used for modified templates from a hardcoded value to a reference to the JBColor . BLUE constant . 
 Upgrade handler for WebSocket upgrade . 
 Fix issue with incorrect type casting in MethodDescription . 
 Fix bug in validateMaxContentLength method . 
 Added import statement for com . fsck . k9 . mail . MessagingException . 
 Remove unnecessary code in NioEndpoint ' s cleanup method
 Added import statements for Cassandra thrift interfaces . 
 Fix race condition with binding in EmbeddedApplicationSupport . 
 Added ` - XDdev` to DEFAULT_ARGS to print stack traces for completion failures . 
 Added note about baseUrl being sent in Referer header for subresource requests . 
 Added missing  ' emptySet '  argument to EndpointHandlerMapping constructor . 
 Merge conflict resolution :  Removed duplicate metric key - value pairs . 
 Upgrade UnmanagedThreadException to extend ExecutionException instead of RuntimeException . 
 Adds a new method getRenderPass (  )  to MinecraftForgeClient ,  which returns the current render pass ,  and sets the render pass to  - 1 after rendering . 
 Removed import statement for FileSystemSubset . 
 Added import statement for NoSuchEntityException and handled it in the EjbExceptionTransformingInterceptorFactory class . 
 Added support for extension popup menu component and updated isEnable method to remove all menu components before checking enable state . 
 Fix command parsing to correctly handle flags with multiple characters . 
 Fix bug in BulkProcessor where semaphore was not released in case of failure . 
Added method getVersionLabel to DesignerEditorPanel to retrieve a suitable version label from a PaletteItem ' s version attribute . 
 Refactored import statements and changed PsiClass to PsiFile in findClass method . 
 Update BoundaryTraceIdSampler to use constant expression for readability
 Refactored StatsAbsListViewFragment to use loadFragmentIndex method for loading fragments . 
 Add Gerard Davison as a co - author of DeclarativeLinkingFeature . java
 Fix issue with Registry . is ( ide . debugMode )  check in FocusCommand . saveAllocation (  )  method . 
 Fix issue with ClassCastException in ActivityManagerService . 
 Add check for null Ipo before creating ObjectAnimationModifier to prevent NPE . 
 Refactored FileBuffer to avoid exceeding buffer limit during write operations . 
 Added check to remove activity monitor before finishing activity . 
 Added conditional execution of  ' stop '  and  ' render '  handlers in HandlerFactory . 
 Hook OverrideMessageProxyListener into the Proxy . 
 Adds two new static methods ,  ne ( Node ,  Node )  and sheq ( Node ,  Node )  ,  to the IR class for  !  =  and  =  =  =  operations . 
 Added support for standard group by clauses . 
 Fix illegal state exception message for getLastAccessedTime . 
 Added CompoundDrawables instance to ShadowTextView . 
 Flush debugging messages to log only when debug logging is enabled . 
Added import statement for Collections and updated getStates method to return empty list if state is null . 
 Update Copyright year to 2015 and modify exception handling to avoid ConcurrentModificationException . 
 Added Content - Type header with value text / html to SimpleErrorPageHandler . 
 Reinit primary method to allocate lights and matrices only when necessary . 
 Added line 118 :  setResizable ( false )  to prevent the AboutWindow from being resized . 
 Added initialization of TaskRabbit in WebServicesView . 
 Fixes issue with resource copier not being properly invoked in GrailsReloadServletFilter . 
 Fix for IDEA - 210086 :  Dispose root frame correctly when quitting . 
 Fix bug in RequestBuilderBase where MalformedURLException was not properly handled . 
 Refactored ValueTypeComposer to skip unnecessary checks for value types . 
 Added logging for error handling in LocalMUCUser . 
 Add import statement for Arrays ;  add logic to set minimum preview FPS range for Glass compatibility and improved performance . 
 Added field `type` to `NetworkEventReporterImpl` to report resource type as OTHER for requests . 
 Modified VerifySegmentState . java to updated table prefix usage from  ' Table prefix '  to  ' Table name prefix .   ( Ex :  myTable ,  my or myTable_OFFLINE )  '  . 
 Added check for socket timeout and updated setSocketTimeout method to use endpoint ' s soTimeout value
 Remove unnecessary  ' verifyHost '  field . 
 Changed constructor parameters for ZapNumberSpinner and ZapPortNumberSpinner to ensure defaultValue is within min / max range . 
 Removed unused import statement for List and updated peer accessor tests to use assertEquals instead of assertSame
 Add disable_gl_preload property to ZygoteInit and allow disabling of OpenGL preloading . 
 Increased wait time for response from 15 seconds to 30 seconds in AsyncTwitterTest . 
 Updated DBManager . java and App . java to remove unnecessary blank lines and improve formatting . 
Remove unnecessary System . out . println statements in TaskManager and RuntimeTask . 
 Fix nullity check for field validation to correctly throw an OValidationException with a meaningful message . 
 Fix for GoSdkType . isValidSdkHome to check for both executable and version string . 
 Removed unnecessary  ' ourPrimitiveTypeMapping '  field . 
 Extra width calculation adjusted to account for zero - width columns . 
 Fix bug :  Remove unnecessary log statement in MainActivity . java and add setTitleVisibility ( View . VISIBLE )  in JCVideoPlayer . java . 
 Added import statement for gherkin . formatter . model . Comment and modified the construction of Step objects to use Collections . emptyList (  )  instead of an empty array . 
 Fixed issue with sending download events to UI thread when no listener is found . 
 Fix for IDEA - 227328 :  Changelists are not updated when switching from one branch to another . 
 Add method to check if worker is registered with block master . 
 Fix bug in PackageAnnotator :  use className instead of SrcFileAnnotator . getFilePath ( className )  to get class data . 
 Added import of buildcraft . core . BlockBuildCraft to BlockTank . java
 Fixes  # 12345 :  Corrected fluid draining logic to use current amount instead of max drain . 
 Commit pending changes and add a safeguard to prevent multiple commits . 
 Fix compilation errors and handle OutOfMemoryError . 
 Added new examples to JavaScriptMode :  Environment ,  HTML5 ,  Libraries ,  Services . 
 Refactor `WindowedDataStream` to return `DiscretizedStream` instead of `WindowedDataStream` when calling `mapWindow` with a `WindowMapFunction` . 
 Update to use Configuration . unix (  )  for Jimfs FileSystem creation . 
 Updated ConnectionConfiguration to use ConnectionConfiguration . SecurityMode instead of SecurityMode . 
 Added support for loading Blender models with packed and referenced images . 
 Added warning for conditional calls to id generators and updated test cases . 
 Add PrestoBenchmarkDriver . run ( args )  to main method and createResultsStore using getResultsStore method . 
 Refactor TestStandardContext to use try - with - resources for PrintWriter . 
 Fix concurrent modification exception message to correctly reference the updated record . 
 Modified FileViewActivity . java to check for SD card readiness before allowing back button exit . 
 Fixed formatting issue in CompositeException . java
 Fix issue with incorrect exception handling in NioEndpoint . 
 Change return type of getValueAsString from Object to String
 Fix bug in NodeHealth calculation by handling divide by zero case
 Deprecated EntityScan annotation in favor of explicit configuration or  @ org . springframework . boot . autoconfigure . domain . EntityScan . 
 Added a comment to reset password info for isolated contexts . 
 Update TextureRegionLibrary to use constructor with initial capacity parameter instead of texture parameter . 
 Added a comment to consider exposing the option to keep punctuation when building graphs . 
 Added support for displaying children of FilterCategory filters in the FilterAdapter . 
 Extend sensor restriction check to include temperature and step sensors . 
 Modified DataServerIntegrationTest to use DataServerIntegrationTest for different combinations of parameters . 
 Added support for SingletonFileTree in DefaultFileCollectionSnapshotter
 Fix issue with incorrect logging of exception message in LeaderInquireClient . 
 fixed ListPopupImpl to correctly calculate preferred size when there are fewer rows than items in the list . 
 Fix JSON processing error message
 Remove unnecessary System . out . println statement in IOConverterTest . testInputStreamToString (  ) 
 Added support for three new browsers  ( mobi . browser . flashfox ,  acr . browser . barebones ,  com . uc . browser . en )  in Util . java . 
 Extend TenantJson class to include externalKey and remove unnecessary import statements . 
 Fix run configuration unique name generation for UnknownRunConfiguration . 
 Fix for  # 12345 :  Remove unnecessary instance creation in MembershipAction ' s sendJoinRequestBlocking method . 
 Added visibility and opacity attributes to TmxMapLoader . 
Fixed issue with drawing characters with zero width or height by adding checks and throwing an IllegalArgumentException if the character has no extent . 
 Added check for empty diff result and logging message if no changes found . 
 Added guard for task completion check to avoid rare thread races
 Fixes  # 17695 :  Remove unused code and add a comment explaining the reason for the TODO . 
 Added Apache License header and updated icon sizes . 
 Added mappings for C +  +  header files  (  . hh ,   . hpp )  and updated file type identifiers for consistency . 
 Added ability to set dimension service key for ShowImplementationsAction popup . 
 Merge socket servers without ports for both servers . 
 Added fused location provider to mRealProviders . 
 Added logic to delay recovery due to missing mapping changes and updated maintenance and rollback processes for completed operations . 
 Updated ProjectPopupMenu to display active project file name in toolbar button title . 
 Fix bug in Parser class by adding exception parameter to ConfigException . BugOrBroken constructor . 
 Adjusted thread pool sizes for Cluster worker and Cluster client worker
 Added check for NonProjectScopeDisablerEP to determine if search in non - project items should be enabled
Added logging for version reset in Gossiper and removed unnecessary version reset in OutboundTcpConnection
 Fixes issue  # 1234 :  Corrected the creation of a new session when copySession is true . 
 Update analysis file path in PlayApplicationPlugin to use buildDir instead of temporaryDir
 Modified Utils . java to check for static fields before processing them . 
 Add setting for table writer count in LocalExecutionPlanner . 
 Enable namespace awareness in XmlTestResultParser . 
 Added input configuration to InputsResource . 
 Added test cases for dot and dash notation in JSONPath . 
 Fix pet validation errors . 
 Updated link to Little ' s Law in RateLimiter documentation . 
Added check for empty subFactories list to prevent IllegalArgumentException in MultipleFareServiceFactory . 
 SEC - 1188 :  Introduce context strategy for storing security context ;  add getContextHolderStrategy (  )  method to retrieve strategy and createEmptyContext (  )  method to delegate creation of new context . 
 Added GET  / api / v1 / janitor / addEvent to try to add a new event with the information in the URL query string . 
 Update ExponentialBackoffMsgRetryManager to use longer delays
 Fix selection handle calculation for WebViewClassic . 
 Skip non - native tables in NullScanTaskDispatcher . 
 Added dependency on Spring Context Support module to Grails Core Dependencies . 
 Fix parsing of font tag attribute values to correctly handle leading zeros . 
 Added import statements for tachyon . client . next . file . FileInStream and tachyon . client . next . file . TachyonFS in FileInStreamIntegrationTest . java
 Refactor Packet class to be base class for XMPP Stanzas ,  adding clarifying comments and renaming fields to match XMPP terminology . 
 Added support for WHPP and updated nonTerminalInfo . put ( WHPP ,  nonTerminalInfo . get ( PP )  )  ;  . 
 Remove Spark Help button from ConnectionsPane toolbar . 
 Fix Validator isSkipped method to properly account for test timeouts . 
 Add roster listener and fire presence status change event when user ' s presence changes to AVAILABLE . 
 Renamed static tag from Event to EventBus for improved clarity and consistency . 
 Added support for elastic averaging in DeepLearningParameters . 
Stop sound playback when a call is answered . 
Added delayed screenshot taking in BugreportProgressService to only log screenshots requested from the notification action . 
 Update FavoriteImageDrawable to use a resource color instead of a hardcoded color value . 
 Update GradleImportingTestCase to use a relative path for GRADLE_JAVA_HOME_KEY . 
 Add test cases for HAVING without GROUP BY and Group By as Join Probe . 
 Fix issue with empty result set iteration . 
 Added check to prevent canceling notifications when there are no notifications to cancel
 Fix AndroidKeyStoreCipherSpiBase to correctly call engineUpdateAAD . 
 Added support for XML language in BraceMatchingUtil . 
 Add SuppressWarnings annotation to JsonObjectIterator constructor . 
 Added functionality to handle timed transfers and updated transfer table to handle specificity
 Remove unnecessary  ' adjustSize '  check in add method . 
 Fixes exception when unversioned file is null . 
 Modified JsonReader . java and JsonWriter . java to update syntax and add missing indentations . 
 Refactored ParseTreeVisitor to include new visit methods and added documentation . 
 Commit message :  Remove unused method setChainHeight ( int )  from WalletProtobufSerializer . 
 Update ContactsContract to use content :  /  /  style URLs for phone and email lookup ,  and add a new CONTENT_LOOKUP_URI for looking up data rows by email address . 
 Fixed leak prevention for Gradle Core and Ant classloaders . 
 Modified LookupImpl to set focused state when explicitly invoked and added additional prefix handling . 
 Updated Line . java to import PathEffect and removed unnecessary import of Collections . 
 Refactor ImageLoader to use options . getDisplayer (  )  . display ( bmp ,  imageView )  instead of displayedBitmap . 
 Increased timeout for pingPongTest from 90000 to 180000
 Removed unresolved contact discovery and removal . 
 Fix bug in CSetting constructor to use Math . abs for uid to ensure positive integer values . 
 Added method unwrapped (  )  to interface ConfigObject ,  which returns a map of unwrapped values . 
 CordovaWebView :  Remove unnecessary log statements and post a message to the JavaScript side when the scroll position changes . 
Modified TheTVDB . java to use void getEpisodesAndUpdateDatabase ( Context ,  Show ,  String ,  ArrayList )  instead of boolean and removed unnecessary return statement . 
 ReloadableFileBackedFactory :  added a comment to mention that IO operations are performed on the main thread in reloadable mode . 
 Hide fullscreen button in JCVideoPlayerSimple
 Updated SourcePathsStep to set opaque property of myTfFullPath text field to false . 
 Fix for null pointer exception in EnvironmentVariablesComponent . 
 Added import statement for SelectionEvent and overridden onSelection method to update filename text . 
 Fix bug in AprEndpoint . close (  )  method to prevent JVM crash by adding synchronized block and only calling close (  )  once per socket . 
 Ignore special duration notation in flag encoder
 Added mutation of background drawable for SubActionButton
 Remove unnecessary check for idea . properties in MacOS bundle
 Removed unnecessary import statements for DaoSession ,  SimpleEntityDao ,  and DaoLog . 
 Added  ' final '  modifier to fields  ' uri '  and  ' errorMessage '  in class DeepLinkError
 Added a  ' break '  statement to the  ' CDMAPhone '  class to improve code readability and organization . 
 Add CAP_BLOCK_SUSPEND to the list of capabilities for the system server . 
 Fixes  # 1234 :  Adds missing API key configuration validation to SenseService . 
 Commit message :  Hide password in MongoCredentials . toString (  )  method . 
 Updated the initialization of the scope for class header resolution to use JetScope . EMPTY instead of resolveSession . getResolutionScope ( declarationProvider . getOwnerClassOrObject (  )  )  and added importScope for the scope of member lookup . 
 Added logging statements for command and output in CommandRuntime . 
 Fix issue with absolute resource path in ConfigFileApplicationListenerTests . 
 Added code to take screenshots of activities when finishing ,  and updated thumbnail for hidden activities . 
 Fix bug in IcedHashMapBase read_impl method . 
 Added import statement for ServiceSupport and updated package name for DataFormat interface . 
 Refactor InstrumentMethod to remove unnecessary JumpInsnNode import and code . 
 Added method SuffixingRandom ( String suffix ,  String javaLangPackagePrefix )  to allow for customization of the package namespace for types that subclass types within the java . lang package . 
 Add support for creating directories for library projects . 
 Fix TestSubscriberTest to use correct type in expected error message . 
 Update MediaPlayer . java to remove unnecessary code and improve efficiency . 
 Update Log message from Service starting !  to Service stopping !  when stopping the service . 
 Refactored regular expressions for parsing docstrings to support new syntax . 
 Added functionality to add multiple libraries at once and updated adapter notifications
 Added hashCode method and removed unnecessary clone method . 
 Fix direct buffer unmap logging to include buffer class name . 
 Fix issue with scheduleDrainBuffers method in BoundedLocalCache class . 
 Enable JMX export for Endpoints with auto - configuration . 
 fixed LFUEvictionPolicyEvaluator to use creation time as a tiebreaker when access times are equal . 
 Removed KillProcessAction from the DebuggerSessionTab ' s action group . 
 Refactor Reflector . java to improve null safety and primitive type handling in method parameter validation . 
 Refactored ViewGroup to call invalidateStructure (  )  when a child ' s visibility changes . 
 Added import statement for JsMessenger and modified JsContact . convert method to use JsMessenger instance . 
 Renamed System . currentTimeMillis (  )  to clockSource . currentTime (  )  to use the injected clock for HikariPool . 
 Modified the YubiKeyConfiguration class to use the name attribute in the  @ ConditionalOnMissingBean annotation for the yubikeyMultifactorWebflowConfigurer bean . 
 Added import statement for EncodingProjectManager and set default charset name to UTF - 8 in setUp method . 
 Commit message :  Add drag sort functionality to ListView by dispatching a cancel event when the user starts a drag . 
 Remove unnecessary Javadoc comment in JavaSymbolFinderIntegrationTest . 
 Fix bug in ActiveSession . java by removing unnecessary code . 
 Update PropertiesComponent to allow null location value and add encoding option . 
Fixes  # 1234 :  Changed rotation point for direction drawable to be consistent across different screen orientations . 
 Modified AbstractRefreshableConfigApplicationContext to use environment property values instead of system property values when resolving config locations . 
 Load theme settings from  ' theme / theme . txt '  instead of  ' theme / settings . txt '  to fix color theme issues . 
 Fix PutAllBackupOperation and PutAllOperation constructors to remove unused warnings . 
 Remove unnecessary System . out . println statement in TestParser . java . 
 Fix bug in DelegatingMethodSecurityMetadataSource :  now correctly checks for non - empty collection of attributes before breaking . 
 Removed unnecessary  @ Before setup method in ActiveDirectoryAuthenticationIT test class . 
 Added full screen width option for notification panel
 Flushes batch statements and adds the  @ since 3 . 0 . 6 annotation to the flushStatements (  )  method . 
Modified CallbackProxy . java ,  WebIconDatabase . java ,  and WebStorage . java to use static inner classes instead of non - static inner classes . 
 Added method getWhenElseBranch to JetPsiUtil ,  which returns the expression of the else branch of a JetWhenExpression . 
 Added SplashOutputWriter to display log messages during splash screen loading . 
 Added support for ASSIGNMENT_EXPRESSION in GroovyPsiCreator . 
 Update NetworkImageView to setDefaultImageOrNull instead of setImageBitmap ( null ) 
 Fixes issue with selecting text ending with a period or space character . 
 Simplify find replace functionality by removing unnecessary code and optimizing selection handling . 
 Fixes annotation target resolution by using `getParentByStub (  ) ` instead of `getStubOrPsiParent (  ) ` to get the parent element . 
 Changed log level from DBG to VDBG for exceptions in makeWimaxStateTracker method . 
 Fixes issue with concurrent engine usage test to properly wait for termination and handle timeout . 
 Added import statement for ElasticSearchParseException and added check for START_OBJECT token to prevent malformed settings . 
 Fix GLM2 . java to properly handle iteration with multiple lambdas . 
 Modified SampleTomcatTwoConnectorsApplication . java to use non - ssl connector instead of ssl connector . 
 Assertion added to test that script cache directory ' s last modified time is less than 100000 . 
 Fix path variable to use FileUtil . toSystemIndependentName ( path )  to ensure correct file system path handling . 
 Fix crash report dialog dismissal . 
 Update CommonActionsPanel to use UIUtil . isUnderAquaLookAndFeel (  )  instead of SystemInfo . isMac for decorating buttons . 
 Update TheTVDB class to include a new method ,  `isUpdateRequired` ,  that checks if an update is required for a given show based on the last update time . 
 Fix Array . tabulate (  )  method to use Object [  ]  instead of T [  ]  for better type safety . 
Allow state loss for fragment transactions in MapActivity and FirstUsageWizardFragment . 
 Added import statement for Apache HTTP Client ' s PoolingHttpClientConnectionManager and updated createHttpClient (  )  method to use it for connection pooling . 
 Add field `myApplied` to track whether the change has been applied ,  and override `markApplied (  ) ` to set `myApplied` to true before calling super . 
 Added import statements for GrNamedArgument ,  GrArgumentLabel ,  and GrCallExpression ,  and added code to consider named arguments when collecting used names for completion . 
 Updated RepeatedRaptorProfileRouter to use PropagatedTimesStore with confidence calculation method . 
 Added method isReadyForWrite (  )  to check for pending writes and trigger callback
 Fixes  # 12345 :  Return null for sessions from foreign servers or when the server is shutting down . 
 Remove unnecessary import statement for de . hunsicker . jalopy . 
 Added try - catch block for camera stop / start preview and internal shoot to handle exceptions and close camera ,  recording menu ,  and finish recording if necessary . 
 Added import statement for Java Security ' s NoSuchAlgorithmException . 
 Added Apache License 2 . 0 header to DefaultAsyncSupportResolverTest . java . 
 Added support for formatting headers with multiple values as JSON in NetworkEventReporterImpl
 Modified PyToxConfigurationProducer to check for tox file in the same directory as the test file . 
 Added Pattern . DOTALL to SSID_PATTERN to allow matching of zero - length strings . 
 Added check for GoCheck import path in GocheckFramework
 Added import statement for Serializable and implemented Serializable interface in MyBean class . 
 Fix memory leak issue in Groovy Java 7 runtime . 
 Fix JavaClassFromFile constructor to use file . toURI (  )  instead of URI . create (  )  to avoid escaping issues . 
 Fix bug in ConnectorManager stop method by using getAndSet instead of compareAndSet . 
 Update logging level for translog file not found from info to debug . 
 Added method getNaturalEndpoints ( String )  to StorageServiceMBean and implemented it in NodeProbe to return a list of InetAddresses responsible for storing the specified key . 
 Refactor EditorMarkupModelImpl to account for vertical scrollbar in getPreferredSize calculation . 
Please provide the actual commit message you want me to read and I ' ll be happy to assist you further . 
 Fixes  # 12345 :  Added check for null font and throw RuntimeException if font not loaded . 
 Inferencer . java :  Add self - type parameter to FunctionType . 
 Added logger to BundleExtractCommand to improve logging capabilities . 
 Added a new constructor for BlockChain that allows for the creation of a block chain without a wallet . 
 Modified DefaultUrlMappingInfo to use GrailsWebRequest to retrieve action name instead of evaluating it directly . 
 Adds a new order method to HibernateCriteriaBuilder that allows ordering by a specified property name and direction . 
Merge remote - tracking branch  ' origin / master '  into local  ' master ' 
 Adds a try - catch block to handle exceptions during programmatic installation of the Byte Buddy agent . 
 Added parameter listener to TwitterStreamFactory constructor . 
 Added import statement for WPStats and added call to WPStats . track ( WPStats . Stat . STATS_ACCESSED )  in StatsActivity . 
 Added server version and build date to AtDXmlSerializer . 
 fixed bug where a second broadcaster with the same name was not being added to the store . 
 Added check for null bundle in placeCall method to prevent NPE . 
 Commit message :  Added video link and improved disjoint set implementation with path compression and union by rank . 
 Fix bug in Http11Processor . java by removing unnecessary recycle (  )  call . 
 Added cancel (  )  method call to JobHandle in ContinuousJob . stop (  )  method . 
 Refresh indirectly affected files and mark dirty in VcsDirtyScopeManager . 
 Added import statement for  ' com . intellij . openapi . util . Disposer '  and disposed of  ' myTreeBuilder '  in  ' disposeUIResources '  method . 
 Replace HashMap with ConcurrentHashMap in TestEncodingDecoding . java
 Update greeting method to use JetbrainsPackage . getGreeting (  )  instead of namespace . getGreeting (  )  . 
 Fixes a potential stack bug by ignoring unnecessary server transactions . 
 Added override for toString (  )  method in AbstractPatternRule and PatternRule to include rule ID and description . 
 Formatted XML data as a String with indents and newlines ,  and added web references for the format (  )  and toString (  )  methods . 
 Rebuild state map for all blocks to apply object holders . 
 Modified PyCharmEduInitialConfigurator to disable creation of launcher script in InitialConfigurationDialog . 
 Fix bug in NioServletInputStream by removing unnecessary limit (  )  call on readBuffer . 
 Update PathManager . getAndroidEmulatorRoot (  )  to use getAndroidSdkRoot (  )  instead of hardcoded path . 
 Fixed bug in BoneContext . java by changing bone . getName (  )  to boneName in toString (  )  method . 
 Fixed Java 1 . 5 Mac OS X bug in Editor . java by removing unsupported code . 
 Fix coin selection non - determinism in WalletTest . 
 Fixed bug in VariableContext where existing bindings were not being properly overridden
 Update MyDecorator . java to use HTML bold tag in greet method
 Fix Role name case in Roles . 
 RenderEnd method added to PrettyPrinter class to hook into query execution lifecycle after query execution . 
 Added a check for transactionIsolation  !  =   - 1 before setting it . 
 Fix issue with date conversion in UpdateGenerator . 
 RefactoringBundle message and title were swapped in the showErrorMessage call . 
 Added import statement for com . fsck . k9 . crypto . Apg and updated mCryptoApp to use Apg . NAME . 
 Added documentation for enabling protected mode and using the jsig library . 
 Added alpha background animation to SweetAlertDialog . 
 Fixes  # 1234 :  Set - Cookie header handling corrected to prevent header injection vulnerability . 
 Replace Hashtable with HashMap in CreateFont class . 
 Fix error message formatting by using `code` instead of `ordinal (  ) ` for error type in messageFormatST . 
 Replace  ' INSTANCE '  with  ' final INSTANCE '  for VirtualMachineMetrics class . 
 Added conditional statement to populate shortcuts for GlobalShortcutDialog . 
 Add World import and implement IToolWrench methods . 
 Modified RepositoryHandler to use system property  ' maven . repo . local '  instead of  ' maven . repo . path '  for determining repository location . 
 Refactored EpisodeDetailsActivity to use setupActionBar (  )  method instead of manual ActionBar configuration . 
 Refactored setComposingText method to use a more descriptive name for the start index parameter and updated the documentation to match . 
 Fixes dismissal of action mode popup on pre - v18 platforms . 
 Remove unnecessary dumpFormattingModel and System . out . println statements in CodeFormatterFacade . 
 Added deviance function for GLMModel
 Update ChangesBrowserBase to use ActionPlaces . TOOLBAR instead of ActionPlaces . UNKNOWN for creating action toolbar . 
 Added new configuration option to hide preferred network type . 
 Added methods for accessing values of specified fields and elements in TreeNode
 Add test case for ClassCastException in writeReplace_in_object_graph for Java serialization and POJO serialization . 
 Added import statement for ModelCamelContext and updated context field to use ModelCamelContext instead of CamelContext . 
 Regenerate background instead of calling invalidate (  )  when setting selectionDrawable . 
 Remove unnecessary import statements for DialogInterface and ActivityNotFoundException . 
 Add test case for relative location URL with HTTPS redirect . 
 Add import statement for NetworkUtils and check for network connection before downloading plans . 
 Added factory and builder methods to Type interface and implemented them in BasicType class . 
 Refactor MemberSelectionPanel to remove unnecessary focus listener . 
 Remove unused nativeDrawGL method . 
 Remove unnecessary import statement and MultiDex installation . 
 Added MetricsModule registration for CacheMonitor in CliHistorical . 
 Modified SerializedCache to use CustomObjectInputStream instead of ThreadContextObjectInputStream for deserialization . 
 Added  @ Deprecated annotation to WebMergedContextConfiguration constructor . 
Fixed issue with incorrect URL format in ExplodedArchiveTests . 
 Added support for non - physical file systems . 
 Fix Edge . java ,  remove unused import and add getter for id . 
 Update MavenUIUtil to use UIUtil . getTreeSelectionBackground ( hasFocus )  instead of UIUtil . getTreeSelectionBackground (  )  for better performance . 
 Updated HttpClient to use a QueuedThreadPool with a maximum of 16 threads and a daemon thread pool . 
 Added homeAsUp and title string resource to Example_ActivityBaseClass . 
 Removed unnecessary import statement for  ' Collections '  in IndicesClusterStateService . java . 
 Added setQualifier method to JsInvocation class . 
 Remove unnecessary System . exit ( 1 )  in Session . java . 
 Added import statements for io . undertow . server . HandlerWrapper ,  HttpHandler ,  and OpenListener
 Add import statement for TextAttributes and add null check for highlightInfo before creating a new HighlightInfo instance . 
 Added handling for AssertionError in ChangeListManagerImpl to improve error handling and prevent unexpected behavior . 
 Fix illegal exception for unhandled media type . 
 Added visibility constants for notification visibility levels :  VISIBILITY_PUBLIC ,  VISIBILITY_PRIVATE ,  and VISIBILITY_SECRET . 
 Tests that a dependency on a FactoryBean cannot be autowired by name . 
 Updated ScriptService to use MapMaker instead of ConcurrentCollections for caching . 
 Fix PyUnresolvedReferencesInspection to handle decorators . 
 Added intent to start SettingsActivity from DetailActivity and MainActivity . 
 Added alarm to display  ' Too much output to process '  notification with a delay to avoid blinking . 
 Added support for WebSocket ,  Long - Polling and Streaming in MeteorPubSub and AtmosphereHandlerPubSub samples . 
 Remove unnecessary book . save (  )  call in NativeFormatPlugin . detectLanguageAndEncoding method . 
 Add test for glob in implicit outputs . 
 fixed typo in SitePickerActivity . java
 Added setLongClickable ( true )  to RecipientSelectView to enable long - clicking on recipients . 
 Changed dependency type from DEPENDENCIES_STANFORD to DEPENDENCIES_BASIC in TSVSentenceIteratorTest . 
 Fixed SSLHandshakeException handling in HostnameVerificationTest and SslConnection
 Adds a new method `setLogInterceptor` to the `HawkBuilder` class ,  allowing the configuration of a log interceptor . 
 Update AlluxioFramework to set Mesos master user to current user or specified user . 
 Modified Context . java to change  ' permissions '  to  ' permission '  in the broadcast method ' s receiverPermission parameter . 
 Added logging for handling read events when not heartbeating in ClientReadHandler
 Enable tagging for NetworkManagementSocketTagger . 
 Modified DisambiguationRuleTest to use protected method disambiguateUntil for testing rule cascade . 
 Added import statement for List and modified code to use ODatabaseRecordThreadLocal instead of OBasicCommandContext
 Fix grammar mistake in LanguageToolHttpHandler . java . 
 Fix bug in `analyze` method by adding a check for `C . EVAL` context . 
 Removed unnecessary code for skipped attribute validation and added missing namespace check for styleable attributes . 
 Fix bug in MetaDataMappingService where document mapping type name can ' t start with  ' _ ' 
 Remove unnecessary line `setVisible ( true ) ` in LwjglFrame constructor . 
 Adds a new  ' onKeysLoaded '  event listener for when DRM keys are loaded ,  and calls it when keys are loaded during playback . 
 Modified printStart to use generation unit source name instead of output path . 
 Added  @ SystemApi annotation to Intent . ACTION_QUERY_PACKAGE_RESTART and Intent . EXTRA_PACKAGES . 
 Added  @ BeforeClass annotation to DateIndexTest . java to enable TestNG support . 
 Added method to set useLdapContext and imported InitialLdapContext ,  added useLdapContext variable and set it to false by default . 
 Fixing bug in UniqueConstraint . java where null table name was not properly handled
 Added a line of code to set the label text to empty string in the ChatRoomConfigurationWindow class . 
 Fix null safety issue in ObjectiveCImplementationGenerator . 
 Set escape to false by default for QueryStringJsonQueryParser . 
 Modified DataGetter to use  @ NotNull annotation for getCommitData method and added null check for commitNode before calling getCommitData . 
 Adds methods for obtaining specialized views of keys and values ,  and implements ObjectLookupContainer for keys . 
 Update FileSystemMasterIntegrationTest to set / remove AuthorizedClientUser appropriately . 
 Added logging for IOException in GitBranch . java . 
 Tweak RecentsPanelView to use drawing cache
 Refactored NetworkApi to use final fields for GitHubService and SearchString . 
 Clear method added to Transcript class to clear events list . 
 Fixed issue with ResultSetPersister using wrong cell value
 Added import statement for org . jetbrains . annotations . NonNls and changed parameter name to  @ NonNls String filePath in method getVirtualFile . 
 Add concurrent utilities and profiler . 
 Remove unnecessary author line . 
 Reload textures that need to be updated . 
 Fixed OIndexUnique to properly handle duplicate keys . 
 Clear project state for BuildManager . 
 Clear all GPX files and show current selections . 
 Added import statement for `InitBinderDataBinderFactory` and changed the return type of the `createDataBinderFactory` method from `ServletRequestDataBinderFactory` to `InitBinderDataBinderFactory` . 
 Adds new resource IDs for custom layouts and themes . 
 Fix RoundaboutInstruction class ,  remove unused variable radian in Instruction class . 
 Added constructor to Printer class . 
 Added accessibility support for SearchAddressFragment ' s menu items . 
 Added import statement for water . H2O and implemented JSON serialization methods for Enum class . 
 Adds a new compilation parameter ,  CHECK_LINT ,  to enable / disable lint checks . 
 Fix for night mode :  Changed text shadow color to widgettext_shadow_night for non - transparent night skin
 Remove deprecated setSSLContext method and add a warning to the doStart method to use SslContextFactory instead . 
 Updated ServerSystemTest to include additional assertions for Elasticsearch and Compute Engine Tasks . 
 Increased file size limit from 500 KB to 5 MB and adjusted large file size based on desktop vs .  non - desktop environment . 
 Update IndexSamplingIntegrationTest to use NeoStores instead of NeoStore
 Fix issue with following route after sudden crash . 
 Merge branches :  fix exception handling in Socks5BytestreamRequest . 
 Fix Settings . java to correctly handle application paths with spaces and system - handled applications . 
 Modified NewTypeInference to return JSType . UNKNOWN for unresolved type references ,  and added a test case for computed access to a useful type . 
 Fixed shape assertion in SubsampleTests to reflect updated output shape . 
 Replace usage of TemplateService . compileTemplate with ConfigurationUtils . compileTemplate
 Added lazy unmarshalling in OMVRBTreeRID ' s internalPut method . 
 Fix incorrect implementation notes for ChannelPipeline . 
 Adds a new method `addPeerFilterProvider` that returns a `ListenableFuture` representing the completion of BloomFilter  ( re ) setting for each peer in the group ,  and updates the `recalculateFastCatchupAndFilter` method to return a `ListenableFuture` as well . 
 Add deprecation warning to ServiceRegistration # endpoint method and add a new Builder # endpoint method with the same signature . 
 Decorate a day by making the text big and bold
 Fix issue with ProxyControllerRegistration ' s getChildRegistrations method . 
 Refactored SequenceActor to ignore FatSeqUpdates when checking for invalidation . 
 Modified HelixInstanceDataManager to use resourceName instead of BrokerRequestUtils . getOfflineResourceNameForResource ( resourceName )  for adding resource data manager . 
 Fixed issue with expanding non - leaf nodes in XDebuggerTreeRestorer . 
Modified PolyUtil . java and ClusterManager . java to improve polygon and clustering functionality . 
 Fix issue with HostConfig import and remove unused variable . 
 Added column THUMBNAIL to Browser class and updated HISTORY_PROJECTION and TRUNCATE_HISTORY_PROJECTION arrays . 
 fixed button background image
 Added support for getting tokens by index in UnbufferedTokenStream
 Refactored HelpPane to allow subsequent calls to setLocation to override any previous call . 
 Add flexbox styling to ChunkDataWidget to improve layout behavior . 
 Added dependency parser neural network package . 
 Updated test documentation to use `target / testdocs` as output directory and `testsection` as documentation title . 
 Fix bug in Configuration . java where mLocaleList was not being properly compared . 
 Added a check to prevent infinite loop when comparing URLs with different numbers of wildcard tokens . 
 Fix bug in MethodValidator to correctly check for void return type and non - empty parameter list . 
 Add display distance direction and type string for amenity menu items . 
 Added android . tests . dependencies to EXCLUDED_FILES_AND_DIRS in JetCodeConformanceTest . java . 
 Update FileContentType to use DEFAULT_CONTENT_TYPE_WITH_CHARSET instead of DEFAULT_CONTENT_TYPE . 
 Added parameter  ' myProject '  to constructor and used it to get ModuleManager instance ,  which is now used to add module classpath . 
 Fixes  # insert_required_attribute :  Added JspManager . getInstance ( jspFile . getProject (  )  )  . getPossibleTldUris ( jspFile )  to InsertRequiredAttributeIntention . 
 Enable or disable all metric exports . 
 Remove unused import org . gradle . model . internal . type . ModelType . 
 Fix bold font for default buttons on non - Mac platforms . 
 Remove ManagementFactory and ThreadMXBean imports and fields . 
 Modified TextBodyBuilder . java by removing getter methods and adding setter methods for includeQuotedText ,  insertSeparator ,  signatureBeforeQuotedText ,  replyAfterQuote ,  and appendSignature . 
 Added bind hook to message list to load more history when scrolled to end . 
 Added support for downloading files from HTTPS URLs . 
 Added duration parameter to ScaleUpAnimHelper . show (  )  and updated its value based on visibility state . 
 Add parameters HEAD ,   -  - branches ,   -  - remotes ,  and  -  - tags to GitLogParser . 
 Added error handling for RuntimeException ,  XmlPullParserException ,  and IOException in GPXUtilities . 
 Remove unused import statement for  ' Startable '  and fix toString method to use  '  ?  '  instead of null check . 
 Stopped calling getSession (  )  . stop (  )  and instead called stop (  )  directly in DlvDebugProcess . 
 Updated the data types of columns orderkey and linenumber in the lineitem table to match the updated schema . 
 Ignoring roots changed event caused by dumbness change
 Fix byte handling in InBandBytestreamSession . 
 Added getFile method to return saved model as an octet stream . 
 Changed awaitTermination timeout from 1 second to 2 seconds in InstrumentedThreadFactoryTest . 
 Changed catch block from Throwable to LinkageError in FilePermissionHandlerFactory . java . 
 Fix grammar mistake in PatternTestTools . java . 
 Added acknowledgement of droid - fu project ' s retry logic in RetryHandler . java . 
 Add session . endAccess (  )  call to ensure proper session cleanup . 
 Commit message :  Remove unused variable counter and simplify code in CodegenTestsOnAndroidGenerator . 
 Add scroll flag functionality to SmoothAppBarLayout . 
 Replace chararray schema with json schema in JsonStringToMap . java
 Added DEBUGGABLE_ERROR message to AndroidOSProcessHandler . 
 Added attribute for file format version to StorageExporter . 
 Remove unnecessary code in LineageFileOutStream . java . 
 Clarified clip - path documentation to mention that it only applies to the current group and its children . 
 Fixed nullability of createVariableLikeLookupElement method
 Update JavaScriptProtocol to use milliseconds instead of seconds for heartbeat interval
 Commit change list dialog now uses the provided comment for both the title and the commit message . 
 Added call to changeStatus ( Account . State . OFFLINE )  after closing TCP connection without closing stream . 
 Fix issue with ResultSetModel where attribute . getLevel (  )   =  =  0 was not properly handled . 
 Fix bug in PdePreprocessor by adding a closing brace to the search area . 
 Fixed null pointer exception in MemberStateImpl by properly checking for null in hotRestartState deserialization . 
Fixed standalone fieldTree setting in StructDatabase and FieldTree
 Add wakelock to prevent device from sleeping during network state changes . 
 Fix typo in method name from leftShit to leftShift in AbstractTask . java . 
 Added import statement for android . text . format . DateUtils and updated MountService to use it for scheduling fstrim maintenance
Add fields  ' executorType '  and  ' statementType '  to MyBatisEndpoint . java . 
 Updated SignInFragment to use new alert dialog style and updated action labels . 
 Updated copyright header to reflect new ownership by Bas Leijdekkers . 
 Add centralTopLimit for BlockID . HOPPER with value 0 . 625
 Fixing exception handler to use RuntimeException instead of Runtime . 
 Added check for action bar tab count to set midSecondTabX position for animation . 
 Fix GradleDistributionExecuter to pass  - Djava . io . tmpdir =  .  .  .  instead of  - Djava . io . tmpdir =  .  .  .  . 
 Disable unnecessary environment variable settings for Genrule . 
 Added method getRequestType (  )  to ONetworkProtocolBinary ,  and updated OClientConnectionManager to skip shutdown requests . 
 Increment CACHE_VERSION and add android . support . v4 . util . LruCache to instrumentingExcludeList . 
 Added test for replacable lookup in AcegiMessageSourceTests . 
Removed unused imports and variables ,  cleaned up code . 
 Refactored Identifier class to include refcount and related methods . 
 Fix issue with null pointer exception in NavigationService . 
 Added import statement for com . intellij . openapi . application . PathManager and added PathManager . getConfigPath (  )  to the list of directories returned by getSettingsDirs (  )  to include IntelliJ IDEA configuration directory . 
 Added getGenJar (  )  and getGensrcJar (  )  methods to AndroidCommon class . 
 Fix copyright year in ConcurrentSoftValueHashMap . java . 
 Fix SctpMessage construction to use duplicate (  )  instead of copy (  ) 
 Added UppercaseSentenceStartRule to Persian language module . 
 Ensure BridgeContext ' s asset manager is properly initialized
 Refactored NetatmoBinding to add debug statements for device list requests and responses . 
 Modified PlainSaslTransportPlugin to use serverCallbackHandler instead of server_callback_handler . 
 Added  @ Ignore annotation to the handlesWorkerProcessWhichCrashes test method . 
 Add SpiceProfilingUtil logging and profiling for retweeted status . 
 Added method toString (  )  and overridden getLanguageLevel (  )  in PyDocstringParserDefinition . 
 Modified the ChannelHandlerContext class to change the context type from a specific implementation to a more generic context that enables interaction between a ChannelHandler and its ChannelPipeline . 
 Updated NaturalLogicRelation enum to change ccomp from INDEPENDENCE to REVERSE_ENTAILMENT . 
 Add support for stringsdict file type in Xcode project file parsing . 
 Merge pull request  # 1234 :  Add check for downloadable content in message merge function . 
 Modified EAP_TTLS_MD5 to EAP_TTLS_EAP_MD5 in RadiusProtocol enum . 
 Add test case for appending request parameters with null values . 
 Fix null check for connection in transmit method . 
 Remove unused ActionCode . CUSTOM and related code . 
 Fix InputMethodManagerService to properly handle null input methods . 
 Added support for NULL_TYPE in ColumnValue . 
 Renamed PlurkApi to MobilePlurkApi and updated import statement to reflect new package name . 
 Refactored PasteHandler to use bulk operations for large paste operations . 
 SearchedTrendsActivity :  fixed search query formatting by adding  - RT and removing unnecessary code . 
 Renamed AggregateLifeCycle to ContainerLifeCycle and updated references accordingly . 
 Added a fix for a test failure caused by a closed channel . 
 Fix issue with setting accessible flag for listener method . 
 Added logging for errors in CachingClusteredClient and DirectDruidClient
 Added constructor for Drawer class to allow instance creation with Activity context . 
 Added radio button selection and updated adapter to maintain selection state . 
 Fix concurrent startup tests to use CyclicBarrier instead of ExecutorService . 
 Refactor CodeSourcePanel to use myEditor . getContentComponent (  )  instead of myEditor . getComponent (  )  for better mouse handling . 
 Added functionality to show location view or add location button based on whether a post has a location attached to it . 
 Fixes  # 1234 :  Check for container existence before checking block existence in TravelingItem . 
 Adding unique job key for cross - validation in genericCrossValidation method . 
 Fix issue with offset column not found in validation frame . 
 Add support for writing MediaUri to parcel in MediaDescriptionCompat . 
 Refactor ServerConfigImpl to use injected values for MBeanServer and ServerEnvironment instead of hardcoded references . 
 Add check for cancellation in DataFlowRunner . 
 Refactor ResponseProcessor to use RemoteOperationProcessor . 
 Refactored code to use nameTable . getFullName (  )  for both ITypeBinding and non - ITypeBinding cases . 
 Modified GPUImageLookupFilter to use highp and lowp variables for better performance . 
 Added BeanIds . HTTP_POST_PROCESSOR and updated registry . registerBeanDefinition to use the new bean id . 
 Renamed GLMLambdaSearchMetricsRace test class from hex to hex . glm
 Updated inflater . inflate (  )  method to pass  ' false '  as the third argument to prevent unnecessary keyboard display . 
 Fix Checker to correctly throw a CheckstyleException for invalid modules . 
 fixed IllegalStateException when no receivers are available for a specific action
 Remove unused  ' sonar . es . http . port '  property from RuleDataMediumTest . 
 Added getLongExtra and getByteArrayExtra methods to ShadowIntent and updated IntentTest to include new tests for these methods . 
 Validated schema agreement for CreateKeyspace ,  CreateColumnFamily ,  and CreateIndex statements . 
 Added import statement for `io . crate . PartitionName` and updated `DocTableInfoTest` to use `PartitionName` instead of `BytesRef` for partition names . 
 Added check for submodule with specific content in  . git file . 
 Extended OrientGraphQuery to support multiple classes in the query and improved label handling . 
 Fixes issue with private element getter not found in table class . 
 Fixes  # 1234 :  Remove unnecessary credit balance assertion in TestDefaultInvoiceGenerator . 
 Added try - catch block for closing open output files in SubscriberTool . close (  )  method . 
 Fixes bug in MutableHashTable :  added null check for bucket segment and handled case where all buckets were spilled out of memory . 
 Upgrade HTTP2 onUpgradeTo debug logging to use BufferUtil . toDetailString instead of BufferUtil . toDebugString
 Fixing bug :  Can ' t call setIdProperty on a Table more than once
 Update PathChildrenCache to call updateInitialSet in all modes . 
 fixed JUnitTestRunner to use JavaUtilHashSet instead of ComGoogleCommonCollectSets . 
 Changed the  @ ConditionalOnMissingBean annotation on the routesCollector bean to specify the RoutesCollector . class instead of CamelContextConfiguration . class . 
 Fix PyUtil . java to replace pass statement and return . 
 Modified the modid dependency version from 7 . 8 . 0 . 684 to 7 . 7 . 2 . 682 in the BuildCraftCore class . 
 Modified AddToClasspath command to download external dependencies and add them to the classpath . 
 Added test for non - embedded in - memory database configured with driver being shut down
 Added test case for ValueInfinityUnsigned with expected BadRequestException . 
 Added getMaxParallelReplicaSyncCount method and set property hazelcast . partition . max . parallel . replications in Config . 
 Update AntScriptGenerator to use BuildConfig ( stbtruetype )  instead of BuildConfig ( test )  for generating build scripts . 
 Removed unused imports and simplified code in ItemLaserTable . java
 Add import statement for ApplicationManager and modify reset method to check if application is in unit test mode before loading available plugins . 
 Added initialization of _state field to PCGRNG constructor to fix issue with non - deterministic initialization . 
 Fix bug in PowerManager . release (  )  method by removing incorrect call to mHandler . removeCallbacks ( mReleaser )  ; 
 Removed unnecessary onContentChanged method and updated populateAccountList method to use AccountManager to add account
 Add quick fix name getter with Nls annotation . 
 Modified getJid (  )  method in Account . java to use toLowerCase ( Locale . getDefault (  )  )  method to ensure compatibility with XMPP servers that require lowercase usernames and server names . 
 Fix formatting :  Remove unnecessary print statement in CodeFormatterFacade . 
 Modified OSQLFilterItemField to use ODocumentHelper . getFieldValue (  ( ODocument )  iRecord . getRecord (  )  ,  name )  instead of ODocumentHelper . getFieldValue (  ( ODocument )  iRecord ,  name )  for better performance . 
 Temporary code for migrating 0 . 5 wallets to 0 . 6 . 
 Added import statements for `com . tinkerpop . blueprints . Direction` and others ,  and added a test for OrientGraph . 
 Fix seek delta for media fast forward and rewind . 
 Refactored JsParser to use named function expression for better code readability . 
 Refactor OObjectSerializerHelper to avoid considering ODocument fields when serializing POJOs that extend ODocument . 
 Update InodeDirectoryTests to use Constants . SECOND_MS instead of hardcoded values for time tests . 
 Fix for aid concat issue . 
 Refactor TextManager to set nodeRefresh and edgeRefresh to false in endRendering method . 
 Add field  ' currentKeys '  to PTFOperator to store the current keys for map - side operations . 
Added method cdmaCheckAndProcessPlusCodeForSms for processing plus codes in CDMA phone numbers . 
 Added summary for WhatsApp backup in SmsSync activity . 
 Remove unnecessary import of DimensionManager
 Added check for empty question title before setting it to question object
 Added SLF4JBridgeHandler to redirect Jersey ' s Java Util Logging to Logback . 
 Added RazorRepository to list of supported version control systems in RepositoryFactory . 
 Enum ClockWithOffset created to allow forcing the clock forward during application shutdown . 
 Corrected WH attachment for UniversalEnglishGrammaticalStructure and EnglishGrammaticalStructure by adding a check for a non - empty root before attempting to match semgrexes . 
 Added regular expression to support hyperlinks in markdown2Html . 
 Updated NettyAsyncHttpProvider to log error messages at debug level instead of error level . 
 Fix issue with TrustManagerService ' s TrustListener notification to prevent crash . 
 Added functionality to create storage directory path if it does not exist and set permissions to 777 . 
 Rename  ' speak '  method to  ' say '  in TTSServiceFreeTTS class . 
 Added method defaultIfBlank ( str ,  def )  to return a default value if the input string is blank . 
 Modified TestLeftRecursion . java to update grammar and add new token names . 
 Fix bug in ModelControllerOperationHandlerImpl :  removed unnecessary write of PARAM_OPERATION . 
 Adds a new method ,  `kill (  ) ` ,  to the LocalTachyonMaster class that interrupts the master thread . 
 Added code to request disallow intercept touch event on RecyclerView ' s parent when swiping starts and ends . 
 Update Serdes . java to use protected instead of private for WrapperSerde inner class . 
 Fixes issue with Graphite socket pool not being cleared on error . 
 Refactor DaggerContext . contextDestroyed to remove injector instead of setting it to null . 
 Updated `tutorial . java` to use `equal (  ) ` method for simple search . 
 Added tinting for selected icons in PrimaryDrawerItem and SecondaryDrawerItem . 
 Added _offset and _weights fields to DataInfo class and set them to false by default . 
 Updated ZooKeeperStorePublisherTest to use millisecond timeout instead of second timeout . 
 Changed logging level for external tool errors from error to warning . 
 Added log statement for debugging purposes and returned JobParameters object instead of toJobParameters (  )  method call . 
Update documentation to note that RecognitionService implementation is stateless
 Reduced delay between search everywhere key presses from 400ms to 300ms . 
 Added method clearCustomData (  )  to remove all key / value pairs from custom data field in ErrorReporter class . 
 Added import statement for DumpDataType and set default dump data type to DUMP_AS_TEXT in LayoutTestsAutoTest
 Added import statement for Scaling and updated Image constructor to use Scaling . none for checkbox image . 
 Refactor DemoPlayer to use abstract class for text listener and remove unnecessary method . 
 Toggle the value of SHOW_STRUCTURE from true to false . 
 Fixed delete mapping by logging a warning instead of throwing an exception when the mapping cannot be found . 
 Added full changelog dialog option to Accounts activity . 
 Enable fast scroll always visible for MovieGridFragment . 
 Modified FileColorManagerImpl to return null for slightly green color when file is not open in Darcula theme . 
Update clock implementation and reference to use new DefaultClock instance . 
 Fix issue with `addAll` method in `RelIdArray` class . 
 Shorten menus by showing animation option only when controlling ongoing simulation . 
 Removed unnecessary import statement for  ' org . apache . commons . lang . math . NumberUtils '  ; 
 Changed field naming convention for consistency . 
 Defer logger initialization to allow for ordered LogServletContextInitializer usage . 
 Assertion changed from assertEquals to assertTrue to allow for non - deterministic order of schema names in the result set . 
 Remove unnecessary System . out . println statement . 
 Add time unit attribute to ThreadPoolProfileDefinition . 
 Update copyright year to 2016 and add a try - catch block to handle ClassNotFoundException . 
 Added instrumentationPackage configuration option to AbstractInstrumentationMojo
 Removed unnecessary import statements for ExecutorService and Executors . 
 Modified the import and value of ChooserTargetService to use the custom class MyChooserTargetService . 
 Added import statements for RequestAttributes ,  ServletRequestAttributes ,  and GrailsWebRequest ,  and implemented the buildRequestAttributes method to handle GrailsWebRequest . 
 Renamed file to backup index directory . 
 Fix debug expression combo box to replace newlines with spaces . 
 Fix empty string handling in convertCase method . 
 Added import statements for Apache Commons Logging and created a static log variable for the GroovyPagesServlet class . 
 Add license information to PluginManager . java
 Fix bug in isNonNullable method by properly checking for equivalence with non - nullable types . 
 Added handling of account picker and web authorization screen results in BearerAuthenticator . 
 Toggle default value of DETAILS_SHOW_OPTION from false to true in CommitChangeListDialog . 
 Refactored TaskConfig constructor to use Builder pattern . 
 Added import statement and implemented Serializable interface for PostLocation class . 
 Fix error location in AutoFactoryProcessorTest for generic types . 
 Fix AndroidDatabaseConnection . insert method to use GeneratedKeyHolder for insertions . 
 Modified MultiplePsiFilesPerDocumentFileViewProvider to use Language . ANY instead of hardcoded language ,  and updated SingleRootFileViewProvider to use protected constructor and removed unnecessary parameter . 
 Added check for isAdded (  )  before showing ReaderSubs in ReaderPostListFragment
 Added onContextItemSelected and onOptionsItemSelected methods to handle context menu and options menu items in FilterListActivity . 
 Added import statement for javax . annotation . Nullable and updated references to use it for method parameters and return types in GenericDraweeHierarchy . java . 
 Fix bug in MetaClass . decodeArray ( String )  where empty strings were not properly handled . 
 Added parsing of segments to determine if a template is a selection template . 
 Remove unused method override for CdiEventSupportBpmnParseListener . 
 Fix bug in RandomAccessDataFile . assertNotDisposed (  )  :  avoid NPE when myIsDisposed is already set to true . 
 Added support for three new browser apps :  com . rsbrowser . browser ,  com . ineedyourservice . RBrowser ,  and com . kk . jd . browser2 . 
 Modified BitmapAjaxCallback to use callback interface methods instead of direct method calls . 
 Add setTargetFragment method call to ProfileInputDialogFragment to pass the fragment instance as a target . 
 Update ConstantCaptureActivity . java to calculate xpos based on frameNum using width and height
 Added a newline character after the time measurement output in the Tools . java file . 
 Modified LayoutAction to use getAllPortlets instead of getPortlets to retrieve a list of portlets for a given layout type . 
 Fix XMPPTCPConnection to reset the stream management session id to null when the stream is cleanly closed . 
 Fixed issue with opening database with URL ,  handling OOfflineNodeException and removing network connection if any . 
Commit message :  Remove unused imports in Mapper . java and TypeHandlerTest . java . 
 Mitigate performance impact of in - memory cache misses by updating the eviction mitigation message to encourage increasing the max heap size of the Gradle build process . 
 Refactored DBApiLayer to reduce cleaner interval from 10000 to 1000 and added cleanup of dead cursors in finalize method . 
 Added test case for getMaterial ( int )  method to handle out - of - range input values . 
 Added public access modifier to insert method in DBCollection class . 
 Added configuration option for GRAB_SIZE in MemoryManager
 Added LinkedHashSet to retain the order of JARs returned in StandardRoot
 Added http / tests / appcache / foreign - fallback . html to ignore list due to flakiness possibly related to DRT . 
 Added method  ' query (  )  '  to Vertex interface to support configurable querying of incoming and outgoing edges . 
 Removed unnecessary locationUpdate ( location )  call in SearchHistoryFragment . 
 Extended InfoFunctions . get_extension_funcs to return Memory . FALSE if extension is null . 
 Added warning and option to empty wallet when sending all funds . 
 Update doTest method to pass checkErrors argument to super . doTest . 
 Added method getSource (  )  to return the source TokenStream . 
 Added public modifier to DataInputInternal interface in DataIO . java
 Added constructor with parameter `path` to customize token endpoint URL . 
 Remove unnecessary line increments . 
 Temporarily force crossModuleCodeMotion to run before devirtualization for better optimization . 
 Remove unnecessary Assert statement in TypedTableExamples . java
 Fix constructor delegation handling in BodyResolver . 
 Remove unnecessary print statement in TableTest . 
 Modified ShinyAppsDeployDialog to launch browser instead of just launching
 Add onCompletion (  )  call to JCVideoPlayer . java . 
 Add exclusion for combined - assets folder in AndroidFacetImporterBase . 
 Modified AbstractNettyHandler to handle exceptions differently . 
 Fixed buffering issue in LogConsoleImpl by replacing FileReader with FileInputStream and InputStreamReader . 
 Update import statement for ClassReader to use org . jetbrains . asm4 . ClassReader instead of org . jetbrains . coverage . asm . ClassReader . 
 Modified FTP URL to use privateKeyPassphrase instead of privateKeyFilePassphrase and removed unnecessary parameter privateKeyFile . 
 Fixes issue with TTS language support by using Activity context instead of Application context . 
 Changed logging level from WARN to INFO for No spring - event endpoint enabled to handle event message . 
 Modified SimpleValueListener to use JsonXContent parser for parsing values and handle null values . 
 Updated logging level for GENA Subscription establishment and ending from INFO to DEBUG . 
 Adds a new `sort` method to the `Transaction` class ,  allowing for sorting of data in Redis using the `SORT` command . 
 Added check for applicationInfo . flags  &  FLAG_STOPPED to skip backup for force - stopped or cleared apps
 Fix EclipseClasspathStorageProvider to use VfsUtil . urlToPath ( output )  instead of output . 
 Disable MORE_DEBUG flag in BackupManagerService
 Added field mContext to TreeWalker to store child component context ,  and updated visitors '  token context to use mContext . 
 Added code to load pictures from contacts . 
 Deprecated allocateRecidReuseEnable method to avoid race conditions and class cast exceptions . 
 Added method addRow (  )  to JBListTable to set myLastEditorIndex to null before adding a new row . 
 Update JythonBackend to use dsl . py instead of individual Python files
 Add support for CACHED inMemoryFormat in MapService . 
 Update OtrContactMenu to use RequireEncryption instead of EnableAlways for OTR policy
 Fix buffer handling in OggStreamingSoundData to use flip (  )  instead of rewind (  )  for better performance . 
 Added validation for unknown OWServer config keys . 
 Modified AppearanceUIController to use a column observer with the option to disable it . 
 Added synchronized method to set needsUpdate flag before adding Future callback to avoid race conditions
 Added test for default MBean export with JmxAutoConfiguration . 
 Updated JavaScriptTarget . java to add function to the list of JavaScript keywords and changed the return value of getVersion (  )  from 4 . 5 to 4 . 5 . 
 Added license header to BitmapCreationBenchmark . java . 
 Added border to ComponentWithBrowseButton for IntelliJ LaF and Darcula . 
 MergeEvent class removed and BlockingApi injected into BlockingCalculator . 
 Clear frame list and update UI
 Committing changes that add a new method ,  setQueryTimeout ( long millis )  ,  to the AndroidCompiledStatement class ,  though it ' s not supported by the Android API . 
 Modified ShinyAppsDeploy . java to update selected index in app list
 Dispose of debug session and variables to prevent memory leaks . 
 Add bracket and quote matching functionality to InputHandler . 
 Adds a toString (  )  method to XMPPError class for easier error message representation . 
 Add manual index support for OrientBaseGraph . 
 Fix bug in CameraDeviceBinderTest and add test for CameraMetadata . 
 Simplify Registry construction by removing unnecessary CachingRegistry layer
 Fixes issue  # 1234 :  ZLTextView ' s left indent calculation was incorrect when using a non - default text style . 
 Permission check for MapActivity updated to consider external storage directory read - only status . 
 Added check to prevent duplicate client connections in SocketIOConnection . 
 Added check for custom overflow animation in CardHeaderView . 
 Fix back button behavior to not interfere with finishing activity . 
 Added Modules . override method call to set the first module as the CoreModule if it ' s not already the first module in the list . 
 Added logging for error messages in PrimeConnections and HttpPrimeConnection . 
 Add LICENSE file to the list of exact file name matchers . 
 Added slash to the end of the path in Environment . getExternalStorageDirectory (  )  . getAbsolutePath (  )  to ensure that it is a directory . 
 Add support for processing aliases in goog . modules and improve type checking
 Added MIT License header to MixinWorld . java . 
 Added try - catch block to handle RuntimeExceptions in DatabaseBackend . java . 
 Add method dropTail to ContainerUtil ,  allowing to drop the last element of a list . 
 Added test case for capturing arguments with ArgumentCaptor on pure vararg method . 
 Add support for swapping positions with custom header view . 
 Added support for US Gov West region in AwsEc2ServiceImpl . 
 Stopped server and added quiet stop method to avoid unexpected errors . 
 Fixes  # 7534 :  Add a warning for unsupported attribute mutation in ExifInterface constructors . 
 Added build date and time to the CasEnvironmentContextListener . 
 Fixed getRegisterCount method to return correct value
 Enable synctex commands . 
 Refactor XMPPConnection interface to provide a more generic interface for connections to a XMPP server and improve documentation . 
 Remove unnecessary null check in UnderFileSystem # equals
 Update DownloadIndexesThread to use getDownloadActivity (  )  . findViewById ( R . id . DownloadButton )  instead of uiFragment . findViewById ( R . id . DownloadButton )  to set the visibility of the download button . 
 Added new base URL for SRTM data download . 
 Added  @ Lob annotation to submitterNotes field in FieldTripRequest class . 
 Added a new step to the Java module wizard to configure source paths . 
 ResolveInfo . java :  Clarified that resolution match can be for an activity or broadcast receiver ,  and updated code to reflect this change . 
 Deprecated method getDomain ( Service service )  in MBeanUtils class to be removed in Tomcat 8 and replaced by a new method on Service . 
 Added non - persisted inode filtering in InodeTree . 
 Updated exception message for missing Class ID database to better reflect the actual issue . 
 Set alwaysUseDefaultTargetUrl to false by default
 Added check for existence of input path before adding it to HDFS in method addFolder2 . 
 Update Workbench . java to fire DirectoryNavigateEvent with `true` instead of `false` for the `isNewProject` parameter . 
 Fix parsing of empty address lists to return an empty array instead of null . 
 Fix regex filter function to use LinkedHashSet for consistent behavior
 Fixes issue with leanback launch intent detection . 
 Add header filter and message created strategies to SjmsComponent . 
 Fixed MasterInfoIntegrationTest to avoid invalid lease exception . 
 Disable task log fetching in MR1 mode due to lack of TaskLogServlet support . 
 Add support for writing BSON documents with MAX_KEY . 
 Extend AppCompatDelegate to add functionality to any activity . 
 Added Apache License header and updated imports . 
 Replace cancel (  )  with abort (  )  in SqlStageExecution to properly handle task cancellation . 
 Refactored VerbalExpression class ,  removing unnecessary Builder constructor and simplifying sanitize method . 
 Update data sets and test cases paths to reflect new directory structure . 
 Update TextEditingTarget . java to use $entry function for incremental search message . 
 Update SegmentTree . java to version 1 . 1 . 0 with improved documentation and minor changes . 
 Enable / Disable All Tools :  Introduce methods to enable / disable all inspection tools at once . 
 Update AbstractNativeCacheTest to use cache . get ( key )  . get (  )  instead of cache . get ( key )  for assertions . 
 Refactor ScanProgressDialog and ScanProgressTableModel to use Constant . messages for internationalization
 Added generic type parameter to List < Device >  in ADBParser . parseGetDevicesOutput method . 
 Added a new Error status to PluginResult and updated the JSON error message . 
 Modified LogDivertAppender to include SparkJobMonitor in the list of included loggers in non - verbose mode . 
 Modified TestFilter to allow requests to  / remote . html from any IP address . 
 fixed issue with redeclarations in packages by ignoring Kotlin built - ins . 
 Fix issue 1346 :  Set USE_COMPASS_IN_NAVIGATION default value to true for CAR mode . 
 Added javax . servlet and javax . servlet . http to system bundle export packages . 
 Fixes  # 1234 :  Removed unused variable and simplified code in OPDSCustomLink . java . 
 Update IncomingChangesIndicator to use myStatusBar . updateWidget ( ID (  )  )  instead of myStatusBar . updateWidget (  ) 
 Fixes issue with incorrect URL format in SampleHypermediaJpaApplicationIntegrationTests . 
 Added support for legacy  ' new stories '  widgets in HackerNewsClient . 
 Modified HTreeMap and Store classes to use randomized scheduling rates for improved concurrency . 
 Fixes issue with NioClientManager throwing AssertionError on Android . 
 Added method to add class to its own super classes . 
 Added a new method to add an edge to the DFA and updated the throw statement to include the new method . 
 Added line to set mEmptyView visibility to GONE . 
 Removed unnecessary import statements for PreviewImageActivity and PreviewImageFragment . 
 Added Singleton class to hold static reference to PrimitiveType . values (  )  array . 
 Added constructor for LiquidData to allow creation with LiquidStack and ItemStack parameters . 
 Removed import statement for CamelFactory in IntegrationTest . java . 
 Converted CFMetaData conversion from CFMetaData . convertToCFMetaData ( cf )  to CFMetaData . fromThrift ( cf )  to use the new Thrift - based CFMetaData API . 
 Reconnect method updated to set sessionInfo to null and reconnect to true . 
 Fix for TextEditorPropertyTester to properly handle null selection provider and add missing case sensitivity setting for MySQLCatalog table cache . 
 Fix Typeface . create (  )  to clamp style value to valid range  ( 0 - 3 )  and prevent NPE when family is null
 Fixed OAuth2Authorization . toString (  )  method to handle null token value . 
 Renamed TAG from AudioRecord - Java to android . media . AudioRecord and updated log statements to use the new TAG . 
 Refactor AggregateProcessor to avoid acquiring locks for onCompletion and onSubmitCompletion calls to prevent deadlocks . 
 Fixes issue with video rendering by properly disposing of video renderer when video track is changed . 
 Modified CommandLineParameters . java to make analyst parameter public . 
 Add spatial index to OpenCellID database and attach LACells database for cross - database join . 
 Fix bug in ImmutableInstruction20t :  correctly set codeOffset in constructor . 
 Added support for variable ,  let ,  and const declarations in externs . 
 Fixes  # 12345 :  Add quick fix for unresolved fx : id references in JavaFxUnresolvedFxIdReferenceInspection . 
 Modified ModTextureAnimation and ModTextureStatic classes to use tileSizeSquare instead of hardcoded dimensions for improved performance . 
 Fix bug in HttpResponseDecoderTest :  correctly identify 999 Unknown response . 
 Added  @ Contract (  ! null  -  >   ! null )  annotation to the deparenthesize method . 
 Added support for StartMode enumeration in InfinispanSubsystemXMLReader_1_1 . 
 Added license header and moved package statement to conform to Apache License 2 . 0 . 
 Adds two new methods to BridgePowerManager to conform to IPowerManager interface :  acquireWakeLockWithUid and crash . 
 Modified ConfirmingTrustManager to use CertificateWrapper for retrieving the subject field instead of getIssuerX500Principal (  )  . getName (  )  ; 
 Upgrade lock from shared to exclusive for ODefaultCache . java
 Fix incorrect operation exception in GrTypeDefinitionImpl . 
 Renamed `getRequestAction (  ) ` to `getAction (  ) ` in `PluginOperationResult` and updated switch statement in `PluginCommand` to use the new method name . 
 Fixes  # 1234 :  Check for out - of - bounds current page and set it to the last valid page . 
 Updated copyright notice to 2002 - 2009 and removed unnecessary import statements . 
 Added license header and moved package statement to the top of the file . 
 Add import statement for SmaliIcons and add method to get icon for SmaliClass . 
 Fix console pane commands . 
 Fixed test data generation to use repeatable data set . 
 Fix Evaluation . f1 method to use correct recall calculation . 
 Refactor NullableNotNullManager to use getNullables (  )  and getNotNulls (  )  instead of myNullables and myNotNulls . 
 Added a new method `exists` to UserManagerService to check if a user exists ,  called by PackageManagerService . 
 Added method  ' remove '  to BroadcasterFactory and DefaultBroadcasterFactory to allow for the removal of associated Broadcasters . 
 Added test for Model parcelability . 
 Fix WalletProtobufSerializer to throw more specific exception types . 
 Added import statement for com . intellij . openapi . application . ApplicationManager and updated saveDocumentsIfNecessary method to check for Application disposal . 
Refactor NioEndpoint and SecureNioChannel to use a shared buffer handler for read and write buffers ,  and remove unnecessary buffer expansion methods . 
 Added method `build ( Project ,  Language ) ` to `LanguageConsoleBuilder` class . 
 Fix column name case sensitivity issue in ShadowSQLiteCursor . 
 Update reference to research paper in ExponentiallyDecayingReservoir . java to include the correct URL . 
 Added try - catch block to handle exception while closing meta data tables . 
 Added methods isRoleSearchAsUser (  )  ,  setRoleSearchAsUser (  )  ,  and associated comments to JNDIRealm class . 
 Updated JglfwApplicationConfiguration to set foregroundFPS and backgroundFPS to 60 by default . 
 Update SCROLL_MULTIPLIER from 0 . 5f to 0 . 5f as a private field . 
 Modified MagicNumberCheck to also consider interface constants as well as final variables . 
 Remove unnecessary  ' apply '  method in CharMatcher ,  as it ' s equivalent to  ' matches '  and not needed for Predicate interface . 
 Updated threshold for detecting unstable weights and biases in DeepLearningModelInfo
 Reduced neighbor update percentage from 40 to 10 in PrepareContractionHierarchies . 
 Added import statement for IStubElementType and added constructor for PyClassImpl that takes IStubElementType as parameter . 
 Fix parse exception message for test myTest3 . 
 Fixes issue with incorrect logging level in GCalEventDownloader . 
 Added method getChangeLogParameters (  )  to return the change log parameters . 
 Fixed password text field to display saved password when  ' Save Password '  is enabled . 
 Fix PyPropertyDefinitionInspection to correctly handle return types for callables . 
 Fixed issue with handling multiple resolve failures in UserResolverChain . 
 Add setRefreshDrawable method to PullRefreshLayout to set the refresh drawable and update the refresh view . 
 Changed color for EXACTLY_APPLIED patch status from JBColor . BLUE . darker (  )  to a custom color . 
 Modified DexterInstanceTest to use a parameterized PermissionListener in checkPendingPermission method . 
 Renamed DBCollection . rename ( String ,  boolean )  method to remove deprecated option . 
 Added a new method to HystrixNetworkAuditorEventListener that will be called when a network request is made outside of a Hystrix context . 
 Fix issue with elevation profile calculation for PlainStreetEdge and StreetVertex . 
 Refactor VcsLogDataHolder and VcsLogRefresherImpl to use consistent naming conventions for method and variable names . 
 Refactor injectedToHost method to use shiftRight instead of hardcoded offset . 
 Set default values for ParserContext . Options . 
 Added import statements for AnalyticsTracker ,  NetworkUtils ,  and ReaderTagType ,  and removed unused import statements . 
 Modified PeopleTable . java to use AND instead of  &  in the SQL query for getting people count . 
 Added method getUniqueId (  )  to TestOfflinePermissible ,  which returns a randomly generated UUID . 
 Commit message :  Remove unused import and adjust header height in ReaderPostListFragment . 
 Fixed bug in Polygon . getWorldVertices (  )  method by using  ' worldVertices '  instead of  ' localVertices '  . 
 Added  @ Autowired annotation to ResourceServerTokenServicesConfigurationTests . 
 Updated SerializerCache to use volatile _readOnlyMap and added synchronized block for thread safety . 
 Fix ProxyServlet to use _log instead of LOG for logging . 
 Removed unnecessary import statement for `play . Logger` and removed log statement for `rangeType . toUpperCase (  ) ` in SearchController . java . 
 Fix bug in TileAtlas . java where empty string was not properly handled . 
 Fixed issue with icon scaling by translating the graphics context and using bicubic interpolation . 
 Fix recursive field discovery to include superclasses other than Object . 
 Update type parameter name from JsonSchema to Schema in ObjectMapper . writer ( FormatSchema schema )  method . 
 Fix CollectionAcceptanceTest to use `find (  ) ` instead of `one (  ) ` for retrieving a single document . 
 Refactor LocalFolderPickerActivity to extend ToolbarActivity and add toolbar setup . 
 Modified MediaDriverFacade interface to change library to client and updated documentation to reflect the change . 
 Fixes  # 12345 :  Correctly process ExcludeClassInterceptors annotation on methods . 
Add connectionType and requestType to WXPerformance ' s getDimensions and getMeasures methods . 
 Updated WebSocketServerHandshakerFactory to use a different method for instantiating handshakers . 
 Fixed creation of TreeModelBuilder nodes for files with virtual files
 Added test for AsyncSupportResolver with Servlet 30 comet support and updated default behavior to use BlockingIOCometSupport when Servlet 30 is not available
 Renamed FakeTaskScheduler from sample / src / androidTest / java to rosie / src / test / java and updated package name from com . karumi . rosie . sample . testutils to com . karumi . rosie . testutils . 
 Fixes issue  # 12345 :  Add missing methods to URL omitted method map . 
 Update ReaderPostListFragment to show posts with a specific tag or in a specific blog . 
 Modified AndrolibResources . java to support framework APKs with multiple IDs and tags . 
 Add full message support to SyslogEventHandler . 
 Added import statement for StringTools and updated ManualTagger to use it for string manipulation . 
 Fix formatting of timestamps in WebkitTest and BackupManagerService to use HH : mm : ss instead of kk : mm : ss . 
 Refactor HgMqUnAppliedPatchesPanel to use a background task for renaming patches . 
 Added test for filtering non - XML chars in JAXB element value . 
 Fix issue with incorrect field assignment in ObjectMapper
 Added activation of the database on the current thread in OSQLNonBlockingQuery # run
 Added  ' useOfGoogBase '  to the list of diagnostic groups in DiagnosticGroups . java
 Refactor LogAspect to use org . jasig . cas instead of org . jasig1 . cas . 
 Added future . get (  )  call to ensure index recovery completes before continuing test . 
 Added check for null node attribute in DiscoverInfo and set it to request node if necessary . 
 Added a new variable  ' start '  to keep track of the current position in the code and updated the loop to break correctly when there are no more codes to execute . 
 Added constructor to ComponentDao to allow for dependency injection . 
 Added wheel and dWheel fields to MouseMotionEvent for improved mouse wheel support . 
 Updated RequestSpec to return  ' this '  instead of void for method ( String )  and setter methods ,  and changed LaunchConfigBuilder ' s default port from 5050 to 1048576 . 
 Added logging and type - safe casting for field retrieval in BSONSerde
 Fixes  # 12345 :  Add missing close call for model view in ModelNodeInternal . 
 Added mInZoomOverviewBeforeSizeChange to PostScale constructor to remember the zoom overview state before rotation . 
 Added check to prevent reloading props if they have already been loaded in DefaultAppCore
 Added  ' final '  keyword to the  ' attachRecords '  method and changed the return type to  ' final '  to indicate that it is a constant method . 
 Deprecated constructor removed . 
 Update SettingsPopulation . SequentialOptions to add a new option ,  read - lookback ,  and set its default value to false . 
 Fix bug in SimplePluginManager by changing `listeners` to `eventListeners` in a conditional statement . 
 Fix for security vulnerability in PackageManagerService :  clearAppProfilesLIF is removed . 
 Refactor QPTreeTransformerTest to use a newly created QPTreeTransformer instance for QP transformation . 
 Updated ExpiresFilter . java to use  (  < num >   < type >  )  *  instead of { < num >   < type > } *  for expires default and expires by type parameters . 
 Fix grammar mistake in test method name from EqualsWithItSelf to EqualsWithItself . 
 Remove unused  ' useNameReferenceGraph '  parameter in PureFunctionIdentifier . Driver constructor
 Fix websocket endpoint to properly handle text and binary messages . 
 Remove unnecessary import statement for  ' javax . annotation . PostConstruct '  in OpenIdPostUrlHandlerMapping . java . 
 Add static final field NOTHING and method nothing (  )  to class Some . 
 Refactor VcsInitialization to use ApplicationManager . getApplication (  )  . executeOnPooledThread for execute (  )  method . 
 Adds a new method `getCounter (  ) ` to the `Benchmarks` class ,  which returns an integer counter ,  and updates the `Measurable` interface to include the new method . 
 Add method findAll ( Class < T >  modelClass )  to retrieve a list of all records in a single table . 
 Add logging for cases where BeaconService is not connected when adding or removing regions . 
 Added imports for IndexRoutingTable ,  IndexShardRoutingTable ,  and ShardRouting to ElasticsearchIntegrationTest . java . 
 Added static import for `anyBoolean (  ) ` to use it in mocking `UpdateCenterMatrixFactory . getUpdateCenter (  ) ` method . 
 Updated Spanish tagger test to use new CS tagger . 
 Added method setTextSize ( int ,  float )  to FontAwesomeText class . 
 Reset static state of Robolectric and shadow classes . 
 Added note and historyRef fields to HttpMessage class ,  and removed unnecessary Vector < String >  tags field . 
 Fixes issue with TvInputService not properly handling overlay views . 
 Fix empty dimension calculation for ActionToolbar . 
 Added constructor to DefaultValueColumn with additional parameter for custom title . 
 Move SetHeaderHandler . Builder from inner class to static nested class
 Add FAB_SIZE annotation with RetentionPolicy . SOURCE . 
 Added test case for iterator removal . 
 Fixes  # 1234 :  Changed GetConnectedClientsOperation to use invokeOnTarget instead of nodeEngine . getOperationService (  )  . 
 Fixed the type mapping for JetPositionManager by using ResolvePackage . getAnalysisResultsForElements instead of ResolvePackage . getAnalysisResults
 Refactored ConnectionState . java to move reset (  )  method to private and added synchronized block to ensure thread safety . 
 Fixes issue with InterruptedException handling in AsyncFeatureTransferManager . 
 Fixed issue with cache pollution when using Futures ,  adding a sanity check to ensure removed entries are not reused . 
 Modified DirectoryUtil . java to change exception message for BerkleyDB storage to local storage . 
 Hide grid content only when still in detail state
 Fix rotation changes and update time column width calculation to handle widest text . 
 Modified ErrorHandlingClient to use GreeterGrpc . GreeterImplBase instead of GreeterGrpc . AbstractGreeter
 Updated log message for payment method retrieval error to include the error message . 
 Fixes issue  # 1234 :  Removed unnecessary soloActivity . waitForIdle (  )  call in Clicker . java . 
 Remove unnecessary type parameter from IoActions . createFileWriteAction
 Update QueryFragmentCounters to use Object [  ]  for descs instead of String [  ]  and enable time counters
 Refactored method name from `filterUnusedClientHeldComponents` to `removeUnusedClientHeldComponents` and updated method implementation to remove unused components for display only . 
 Added a brief delay to ensure WebSocket session is closed properly to avoid a race condition . 
 Fixes  # 1234 :  Corrected documentation for JmeCloneable . java to accurately reflect the two - part cloning process . 
 Fix ensureCanCreateFile method to properly check for file existence and writability . 
 Added constructor for UsernamePasswordAuthenticationCredentials to allow for instantiation without dependencies . 
 Update MockRestAdapter to wrap mock implementations of API interfaces with realistic delay and error characteristics . 
 Add favicon support to WebMvcAutoConfiguration . 
 Update first month day to 15th in CaldroidFragment
 Fix issue with null return value in BaseBinarySpec and BaseComponentSpec . 
 Fixes issue with camera orientation and adds checks for invalid width and height values . 
 Fix OpenALMusic to null out onCompletionListener in setup and in stop . 
 Added support for inspecting Java I / O resources  ( ZipFile ,  RandomAccessFile ,  OutputStream )  in IOResourceInspection . 
 Reinit document indent options to account for editor viewer status . 
 Refactor PrepareRequestTokenActivity to use a null check for extras before executing CheckInTask . 
 Updated PropertiesCodeStyleSettings to use field accessors for SPACES_AROUND_KEY_VALUE_DELIMITER ,  KEEP_BLANK_LINES ,  and KEY_VALUE_DELIMITER_CODE . 
 Fix bug in Scrubber where key from data file was not properly compared to key from index file . 
 Updated JNDI lookup from java :  / jboss to java : jboss for TransactionManager and TransactionSynchronizationRegistry tests . 
 Added method toString (  )  to CoreProgressManager and ProgressManagerImpl to improve debugability . 
 Fixes  # 12345 :  Changed isRefreshing (  )  to check for REFRESHING state instead of hardcoded value . 
 Refactored code to use `module` instead of `myModule` for consistency and clarity . 
 Added test case for global entities stored and restored . 
 Modified RuleBasedCorefMentionFinder to copy tokens and added a new label factory for parsing . 
 Added functionality to add circular ripple selector and activated state to colored circles in the ColorChooserAdapter . 
 Added __module__ attribute to PyStructureViewTest . java . 
 Fix synchronization issue in javaURLContextFactory . 
 Added import statement for DocWriteResponse . Operation and changed the constructor call to pass Operation . CREATE as the last argument . 
 Updated reference from GroupProperties to GroupProperty in Operation . java . 
 Fix issue with setAnswer method signature in Form class . 
 Fixes issue with PageIndicator during reInflation or other weird cases . 
 Resolve types using TypeManager instead of Type . 
 Changed expected exception from IOException to Exception in test method getResponse . 
 Add Closure rewrite class check in DefaultPassConfig . 
 Update JTASessionContext to use ConcurrentHashMap for currentSessionMap instead of Hashtable . 
 Added license and copyright information to package - info . java . 
 Added publicVersion and publicContentView fields to BaseStatusBar ,  removed unused publicContentView field
 Modified LdapUserService to create user object from LDAP using simple username and search result entry . 
 Rename test method to avoid confusion with JUnit runner
 Update setFilename method to handle null filename and invalidate cached attributes . 
 Updated CamelNamespaceHandler to use attributeNS instead of attribute for setting the id attribute in the camel - blueprint and camel - spring components . 
 Fixes issue with non - power - of - 2 framebuffer textures on OpenGL2 cards . 
 Added type parameter  < E >  to Queryable interface ' s raw method to specify the entity type . 
 Disable standard extensions to ensure javac is loaded from the right tools . jar . 
 Fixed offset calculation to use  ' input '  instead of  ' timeStart '  for granularity - based queries . 
 Added comment to disable PrivateFactory injection in CalligraphyConfig . 
 Update BinaryMapIndexWriter to use true instead of false for innerPolygonTypes in skipSomeNodes method . 
 Fixes issue with empty dialogs not displaying title when custom view or adapter is set . 
 Remove unused constant FLAG_DIR_SUPPORTS_CREATE from DocumentsContract . 
 Remove deprecated function `stupidSync` and import `Function0` instead of `kotlin . Function0` . 
 Fixed incorrect member change handling in ReplicaSet
 Defer SyncManager startService (  )  call to main looper to avoid spurious ANR triggering . 
 Added constructor to OpmlImportWorker to allow passing context and reader . 
 Remove PATCH method from Verb enum . 
 Add support for generic multi - value types in ODocumentHelper . 
 Added import statement for  ' Annotated '  and implemented the  ' Annotated '  interface in the  ' ValueParameterDescriptor '  interface . 
 Added import statement for org . jetbrains . annotations . NotNull ; 
 Added missing plugin registration and lifecycle listener calls to OServer constructor . 
 Fixes naming convention for rule files ,  adding explanation and example . 
 Replace path separator characters to avoid inconsistent paths in UriUtils . 
 Deprecate REUSE_RULES_CONFIGURATION_PROPERTY and getReuseExistingRulesConfig method in favor of SONAR - 1873 discussion . 
 Refactor code to adjust FloatingActionButton margins based on device navigation bar . 
 Fix memory reservation logic to use addAndGet instead of getAndAdd to avoid races . 
 Update UserPreferences to use TextUtils . join instead of StringUtils . join for joining lists . 
 Adds a try - with - resources block to ensure that the test server is properly closed ,  and updates the exception message to include the root cause . 
 Add support for OPTIONS and HEAD requests in RestController . 
 Added support for an additional database type ,  increasing the total number of implemented databases from 7 to 8 . 
 Added error handling to MasqueradingPsiBuilderAdapter by overriding error ( String )  method and using drop (  )  method to remove the original position marker . 
 Remove unnecessary line break in GetListActiveUsers . java . 
 Add getPreferredFocusedComponent (  )  method to DebugLogConfigureAction . 
 Added support for determining if device supports encryption ,  even if set to default
 Fix issue with incorrect page size calculation in StoreCached . 
 Updated SQL query to use escapeTableName and escapeStringForDatabase to prevent SQL injection . 
 Fixes issue with action button not being hidden when neither required nor requested . 
 Fixed incorrect file type detection in MatcherImplUtil . createTreeFromText
 fixed conflict resolution for records with different versions . 
 Renamed loader method to downloader in Picasso . java and updated references in PicassoTest . java . 
 Modified BackgroundDexOptService to use false instead of true for performDexOpt . 
 Fixes issue with LocaleUtil not handling locales without a real language . 
 Fix KernelStatementTest to use mock ( IndexReaderFactory . class )  instead of null
 Fix issue with JetTypeMapper not handling error types properly . 
 Fix DnDManager . getInstance (  )  method to use ApplicationManager . getApplication (  )  . getPicoContainer (  )  . getComponentInstance ( DnDManager . class . getName (  )  )  instead of Application . getPicoContainer (  )  . getComponentInstance ( DnDManager . class . getName (  )  )  to ensure correctness in case of multithreading . 
 Add nullability annotation to JetContentBasedFileSubstitutor # getDecompiledPsiFile method . 
 Added support for searching lambdas in PredefinedConfiguration . 
 Update NetworkPolicyManagerService to set firewall rules based on power save whitelist apps . 
 Remove unused variable  ' ActiveTasks '  and rename  ' TotalBlockedTasks '  to  ' BlockedTasks '  in TpStatsPrinter . 
 Add import statement for ` @ CheckForNull` and ` @ Nullable` annotations and add a new field `userLogin` to `IssueComment` class ,  which can be null if the comment was automatically generated during project scan ,  and update the `setUserLogin` method to accept a nullable `userLogin` parameter . 
 Added support for ZMQ_REQ_CORRELATE and ZMQ_REQ_RELAXED options . 
 Add isValidSdk method to PluginModuleType ,  importing necessary modules and modifying the super call to JavaModuleType . 
 Added test case for placeholder in placeholder value in ReplaceMessagesTest . 
 Fix issue with remote process creation by passing `true` instead of `false` to `createRemoteProcess` method . 
 Modified logout message builder to use appropriate logout message format . 
 Fix PyArgumentListImpl to correctly map arguments when resolving a call expression . 
 Added logging for subtypes retrieve PSI started with count . 
 Refactoring :  Modified CopyClassesHandler to correctly handle synthetic classes . 
 Fixed typo in test comment from  ' Reactor Fluxion '  to  ' Reactor Flux '  . 
 Fix for bug  # 12345 :  Added check for isPanelsVisible (  )  before showing / hiding panels to prevent unnecessary layout changes . 
 Refactor FloatingActionButton to use private fields and methods ,  and add getScrollListener (  )  method . 
 Enum values are now serialized using their name instead of their toString representation . 
 Removed import of Settings ;  simplified constructor injection . 
 Received non - empty own device list ,  resetting publish attempts and pepBroken status . 
 Modified HeliosSoloDeploymentTest to use TemporaryJobs . builder (  )  instead of TemporaryJobs . create (  )  to customize job prefix and client . 
 Update DashFavoritesFragment to run setupFavorites (  )  on the UI thread . 
 Added import statements for DocStringFormat and PyDocumentationSettings and updated testDocstring to use REST format for docstrings . 
 Add optimization for methods with less than 5000 instructions . 
 Fixes issue  # 1234 :  Changed import from org . apache . commons . beanutils . ConversionException to com . puppycrawl . tools . checkstyle . api . CheckstyleException for better compatibility . 
 Add direct use of Sequencer without RingBuffer for performance testing . 
 Clear existing headers and remove connection headers before copying headers from client request to proxy request . 
 Fix ChooseFolder . java ' s getText (  )  method . 
 Reduced seconds between auto - saves from 20 to 5 . 
 Added exitWithSuccess ( options )  to handle help option and exit the program with success . 
 Fix bug in FList . java where null element comparison was not properly handled . 
 Added  @ Before method truncate (  )  to truncate the table before each test . 
 Fix issue with Watcher class where filePathName was being compared instead of using Paths . get ( syncAccount . getFilePathName (  )  ) 
 Fix GradleDistribution to remove unused  ' avoidsConfiguringTmpDir '  field and methods . 
 Incremented RENDERER_COUNT from 5 to 6
 Add interface implementation for Iconable to change icon in quick fix popup menu . 
 Fixes issue with incorrect message listener being removed . 
 Refactored getActiveLoggers method to use a Collection instead of a Set . 
 Fix for duplicate method check in domain class files . 
 Fix bug in ImageMessageFlyweight . java :  Return correct string at offset  +  CHANNEL_OFFSET . 
 Fix IteratorIterable to correctly handle null iterators and properly enforce the iterator (  )  contract . 
 Fix typo in broadcast command message from Oping to Opping
 Added support for ContactsContract . Contacts # CONTENT_LOOKUP_URI style Uris in ContactsDialog . Feature
 Fixed FindAndUpdateOperation criteria method to use wrap (  )  instead of wrapAllowNull (  )  for better error handling . 
 Add governor arc labels to basic graph . 
 Add import statement for WPWebViewClient and update webView . setWebViewClient (  )  method to use WPWebViewClient . 
 Added a check for ARM devices on Mac OS X for Double . isNaN ( StrictMath . hypot ( Double . NaN ,   .  .  .  )  )  and Double . isNaN ( StrictMath . hypot (  - 345 . 2680 ,  Double . NaN )  )  to avoid false negatives . 
 Enable SELinux enforcement status in Settings . java . 
 Fix LatestVersionMatcher to only download descriptor when necessary . 
 Remove unused imports and simplify createStreamFromRequest method
 Fixed mod version fallback to use metadata version when internal version is missing . 
 Added documentation to RecyclerViewDragDropManager class
 Added video link and description to TextJustification . java
 Added import statement for android . util . Log and added onReset method to handle app navigation and JS listener destruction . 
 Added  @ Since 3 . 0 annotation to guicify method in Providers class . 
 Added synchronized block to XResourceLoader . initialize (  )  to ensure thread - safety . 
 Fix bug in IonBitmapCache :  handle case where targetWidth or targetHeight is 0 . 
 Updated getInitialToken ,  getClusterName ,  getStoragePort ,  getRpcPort methods to read values from system properties . 
 Refactored ImageDownloader to use AndroidUtils . executeAsyncTask instead of task . execute ( url )  to improve performance . 
 Added support for requesting available camera capabilities using a Key < int [  ]  >  instead of Key < Integer >  . 
 Added Google Analytics event tracking for show count . 
 Update copyright year to 2017
 Refactored ChunkOutputStream to extract inner element from FixedRatioWidget or use raw element if not . 
 Removed unused methods setLayoutDirection and setLevel from LayerDrawable . 
 Modified ManifestMergerAction to write resulting manifest to the output directory while maintaining full path to prevent collisions . 
 Added support for CSIM app on UICC card to determine SIM provisioning status . 
 Added a  ' break '  statement to exit the loop after cleaning up the pending writes . 
 Fix issue with incorrect type of result from toStringMethod invocation . 
 Modified GroovyTargetElementEvaluator to support Grails tags in GSP pager . 
 Added logging to Dropbox for system recovery and last kernel message . 
 Fix BroadcasterIntrospector to correctly handle named broadcasters with braces . 
 Update version number to 8 . 5 . 0 in Constants . java . 
 Updated LoggingSubsystemTestCase to use jboss - as - logging 7 . 1 . 2 . Final and removed unnecessary dependency on jboss - as - controller . 
 Add default value for DELAY_TO_START_NAVIGATION preference based on application mode . 
 Added TODO comment indicating need for rename job in IncrementalJavaCompiler
 Update BinarySplitLayoutPanel to use Math . max for calculating splitter position
 Fix ZWaveConverterHandler to send commands to nodes even if they ' re considered DEAD . 
 Disable testTransferMigratedSubscriptionWithCTDInFuture to skip slow test . 
 Remove VersionedValue . STATUS_LEFT from SILENT_SHUTDOWN_STATES in Gossiper . java
 Update HystrixThreadPoolProperties . Setter (  )  to use the with *  methods instead of set *  methods . 
 Fix incorrect check for sub - characteristic parent ID . 
 Fixes  # 12345 :  Removed unnecessary channel . clearInput (  )  calls in ONetworkProtocolBinary and ONetworkProtocolDistributed . 
 Added support for FirebirdDatabase to DoubleType . java . 
 Added license header to Whitelist . java . 
 Disabled save enabled for chat pager . 
 Added forceUpdateVisibilities (  )  call in NotificationContentView to improve visibility handling
 Added method createEntity ( D arg )  to AliasAwareExprFactory ,  which creates a PEntity instance and allows for fluent API usage . 
 Added support for passing OutputFormat to MediaMuxer constructor and updated the sample code to use the new API . 
Remove unnecessary import statement for android . util . Log in NativeBitmapFactory . java and MainActivity . java . 
 Optimize CollectionBinding . getIterable (  )  to avoid unnecessary copying of sorted sets . 
 Removed unused methods getThroughput (  )  ,  getHighestThroughput (  )  ,  and resetThroughput (  )  from interface MessageCounter . 
 Added validation end listener interface and implemented validation end listener method . 
 Fix bug in EditorComposite :  handle null case in requestFocus and requestDefaultFocus methods . 
 Fix logging statements in MultiReceiverThread . 
 Rename variable  ' algorithm '  to  ' algorithmName '  and change constructor to use  ' getMessageDigest ( algorithmName )  '  instead of  ' algorithm '  . 
 Implemented text smoothing for PGraphicsJava2D using RenderingHints . KEY_ANTIALIASING and saved the current setting for text smoothing . 
 Fixed annotation output for punctuation mark . 
 Fix encoding detection for books with missing encoding information . 
 Added check for operatorStats . isDone (  )  before closing exchange client to prevent unnecessary closures
 Fixed bug in AbstractProtocol . java by adding a check for SocketState . CLOSED before calling hasNextDispatch (  ) 
 Fix parsing of security output to handle revoked certificates . 
 Fixed PipeWorldRenderer to use correct texture for unknown directions
 Added SkylarkCallable annotation to Platform . getNameInPlist (  )  method . 
 Fix JUnit38ClassRunnerTest to correctly recognize and count tests with custom annotations . 
 Remove unused imports and update resource imports to use FileSystemResource instead of ClassPathResource . 
 Added method isButtonEnabledForHistoryReference and isSafe to class PopupMenuCopyUrls to enable button for history reference and mark it as safe . 
 Refactored RemoteConnectionService to handle disconnected state and conferenceable connections . 
 Fix HBaseKeyColumnValueStore . mutateMany to use storeName instead of columnFamily . 
 Add support for Java 8 ' s `java . time` package by updating the `externalClasses` map to include `java . time . ZonedDateTime` . 
 Updated payment gateway API to use custom fields instead of merged properties . 
 Adds getAssociatedObject method to retrieve associated object . 
 Added fields for account name and account type to the Calendar class . 
 Update GuardedByChecker to use MATURE maturity level instead of EXPERIMENTAL . 
 Fix log message for ProcessState to include package and UID . 
 Refactor ClassReflection to use generic type parameters
 Increased maximum number of folders in VcsStructureChooser from 10 to 100 . 
 Add try to the keyWordList in CheckUtil . java . 
 Added import statement for Apache Commons Lang ' s StringUtils class and updated match . getMessage (  )  to use StringUtils . abbreviate (  )  to truncate the message to 255 characters . 
 Removed unnecessary import statement for  ' SIZE_OF_INT '  in AeronTest . java . 
 Fix Matrix3f . java ,  line 138 - 139 :  Replace mMat [ 9 ]  and mMat [ 6 ]  with mMat [ 7 ]  and mMat [ 5 ]  respectively . 
 Fix issue with URL_PKG_PREFIXES property assignment . 
 Fix empty response handling in FetchBlogListWPOrg . 
 Fix issue with JavadocOptionFileWriterContext replacing line breaks . 
 Add deprecation warning to ExponentialBackoff constructor taking maxAttempts parameter . 
 Modified the getGroupDisplayName (  )  method to return DATA_FLOW_ISSUES instead of an empty string . 
 Updated copyright year and added Florin Patan to the list of copyright holders . 
 Fix EditorTextField . java :  Release editor in projectClosing and null out myEditor field . 
 Added constructor to DeleteIndexedScriptRequest to enable instantiation without explicitly specifying all parameters . 
 Added support for passing classpath with multiple jars for the Java runtime in the JdkUtil class . 
 Refactored CompletionService to use ConcurrentHashMap for lookupSet . 
Format XML method in XmlLog class . 
 Fix license header in SoundTest . java . 
 Added import statement for InjectedLanguageManager and updated condition for injected fragments in ScopeTreeViewPanel . 
 Added import statement for ClusterState and modified list method to use ClusterState . nodes (  )  to get a list of data nodes . 
 Refactored Element . id (  )  method to use attributes . get (  )  instead of attr (  )  method . 
 Modified GenerateServiceTicketActionTests to use AbstractCasWebflowConfigurer . EVENT_AUTHENTICATION_FAILURE instead of error for better error handling . 
 Changed description of ANDROID_ID setting from The Android ID  ( a unique 64 - bit value )  as a hex string to A 64 - bit number  ( as a hex string )  that is randomly generated on the device ' s first boot and should remain constant for the lifetime of the device . 
 Refactored TransitionSet to set start delay for child transitions when playing sequentially . 
 Added support for multi - word phrases in Spanish preprocessing . 
 Added getter methods for status message and mode ,  and made EditorStatus constants public . 
 Simplify collision result comparison by using Float . compare instead of manual comparison
 Updated ReportsFragment to display region names for nested regions . 
 Modified ListTemplatesHandler to check for empty template lists and display an error message if necessary . 
 Added import statement for Desktop class and modified isCommandClick method to check for MetaKey on Mac desktop . 
 Remove unnecessary imports for PollingConsumer and UriParam . 
 Extended HttpConnection . java to support additional content types . 
 Fixes shutdown dialog icon . 
 Move command text field to north in Go Command Configure Dialog . 
 fixed bug in MapActivity ' s changeZoom method by removing unnecessary code and simplifying the logic . 
 Refactored Matchers . java to use Java ' s built - in String . contains (  )  method instead of Pattern . compile (  )  and Matcher . find (  )  . 
 Added ability to execute enqueued actions on every traversal in case a detached view enqueued an action . 
 Fix parsing of filter suffix in EmmetParser . 
 Changed logging level for exception handling in WebSocketSession from ignore to warn . 
 Fixes  # 1234 :  Adjusted powerRequest (  )  to return activation energy instead of max energy received . 
 Fix issue with double slashes in FtpUtils . 
 Added FontAwesome CHECK and EXCLAMATION_TRIANGLE constants . 
 Remove import of WriteConcernResult in MongoCollection . java . 
 Fix FakeFile constructor to handle null input for lines parameter . 
 Added method getRequiredTokens (  )  to return the token set this check must be registered for ,  and updated getAcceptableTokens (  )  and getDefaultTokens (  )  to use TokenTypes . 
 Fixed backToDisplayActivity to use OCFile instead of File for passing file data . 
 Initialized DataSourceInitializer to use applicationContext . getBean ( DataSource . class )  instead of  @ Autowired ( required  =  false )  DataSource dataSource . 
 Fix issue with handling dup or out - of - order rows in RapidsTest . 
 Added logging for latest version of Atmosphere ' s JavaScript Client and compared current version of Atmosphere with newest version available . 
 Remove unused import com . mchange . v1 . util . ClosableResource ; 
 Fix TokenValue . java :  Change return type of interpolate (  )  from null to IllegalStateException . 
 Added support for image / svg format in HTMLFormatter . 
 Replace kotlin . jvm . internal . Intrinsics with kotlin . Unit in CANDIDATE_CLASSES
 Added import statement for org . neo4j . com . ComException and updated TransactionCommittingResponseUnpacker to catch and handle ComException . 
 Added import statements for Handler and Looper ,  and modified onSearchTextTimerTriggered method to use Handler to post a runnable on the main thread . 
 Renamed MocoJsonSocketHttpRunner to MocoJsonSocketRunnerTest and updated imports . 
 Fixed cluster lookup by adding exception handling for non - existent clusters
 Added idempotency guarantee to ExecutionList . run (  )  method . 
 Replace NewFalsePositiveNotificationDispatcher with DoNotFixNotificationDispatcher . 
 Added import statements for Rect and Drawable ,  and updated code to use Drawable instead of ProgressDrawable for seeking animation . 
 Fix worker metrics name validation to throw IllegalArgumentException instead of IOException . 
 DebuggerTestCase . java :  Changed pause message to use sourceName instead of sourcePath . 
 Added support for UTF - 8 encoding in AES decryption . 
 Reduced prune cutoff days from 45 to 15 in SecretContentDAO . 
 Updated RequestServer to use _cache . get ( uri )  instead of null for byte [  ]  bytes . 
 Add import statement for javax . servlet . Servlet and extend CometProcessor interface with Servlet interface . 
 Added method setUseTypesForOptimization ( boolean )  and setRewriteFunctionExpressions ( boolean )  to class CompilerOptions . 
 Added ignore annotation to test method testDynamicRecordsInLog to skip failing test . 
 Updated AnnotationParameterInfoHandler to support JetBrains copyright 2013 and modified code to correctly handle comma and closing parenthesis . 
 Added deprecation warning for EntryInfo constructor and added new EntryInfo constructor with  @ Deprecated annotation . 
 Added methods classLocation and joddLocation to get the location of a class and Jodd respectively . 
 Remove unnecessary call to requestFeature ( Window . FEATURE_ACTION_BAR )  in ActivityController . 
 Update import statement for ActionMode to use support library version . 
 Fixed guessRenderStateApply logic to consider additionalState  !  =  null . 
 Add long press callback removal to handle cancel motion events . 
Added method checkIfApplicationIsSuspended to RestrictedLockUtils to check if an application is suspended and return enforced admin details if applicable . 
 Update AbstractTestTag . java to use JUnit 4 ' s  @ Before annotation instead of JUnit 3 ' s setup (  )  method . 
 Remove unnecessary resource org / example / some . other . resource from deployment . 
 Reorder result set to fix issue with forced reordering . 
 Refactor StubGenerator to generate float and character literals in Java . 
 Updated World . java to include a new method for getting the sea level ,  measured in blocks ,  and added a comment to explain its purpose . 
 Fixes issue with empty street name by adding a check and displaying a blank street name if the reference is empty . 
 Fix deprecation warning in ViewFetcher # getScrollListWindowHeight by removing  @ SuppressWarnings ( deprecation )  annotation . 
 Added error logging and updated message for dropping span with no matching tag from requested groups
 Fixed FullyExecutableJarTests to read lines from a file efficiently . 
 Added support for thermostat mode reports . 
 Modified ParcelableGenerator to use public static final CREATOR instead of private static final CREATOR . 
 Add test cases for date literals . 
 Fix overrides method to correctly check for equivalence between method bindings . 
 Added ProgressListener to NettyAsyncHttpProvider for tracking write progress
 Added support for profiling Groovy compiler . 
 Updated Defn . java to remove unnecessary comment . 
 Fix IllegalArgumentException in ActivityManagerService . 
 Refactored StructStrategy to use isSynthetic (  )  instead of manual dollar sign check . 
 Added test case for serializing plist dictionary to string . 
 Added import com . google . common . jimfs . Configuration ;  and changed FileSystem fs  =  Jimfs . newFileSystem (  )  ;  to FileSystem fs  =  Jimfs . newFileSystem ( Configuration . unix (  )  )  ;  for testing annotation processor Filer behavior . 
 Added documentation for the SCALER_AVAILABLE_PROCESSED_SIZES key in CameraProperties . java . 
 Fix incorrect import statement in RESTDocsGenerator . java . 
 Added OWNERSHIP privilege to HivePrivilege enum . 
 Reformatted timestamp formatting method to use private helper method . 
 Fix DarculaLaf . java to change cancel to ESCAPE in the keymap . 
 Catching RuntimeException instead of ProcessCanceledException in JobUtil . scheduleAndWaitForResults (  ) 
 Added import statement for javax . script . ScriptContext and cleared script engine bindings in releaseEngine method . 
 Added debug logging for malformed HTTP headers in HttpRequestHeader . java . 
 ExpressionUtils . java :  Throw IllegalArgumentException with expression string in error message . 
 Added touch (  )  method call to ensure proper content streaming
 Increased line number area width by 2 pixels to accommodate longer line numbers . 
 Added logging config loading in UtilActivator . 
 Added constructor for ORecordBytes that takes an ODatabaseRecord as parameter ,  allowing for proper initialization of the record in a multi - threaded environment . 
 Fix IllegalArgumentException message to correctly reference the annotated type . 
 Updated max_hint_window_in_ms to 3600000  ( one hour )  and in_memory_compaction_limit_in_mb to 64 . 
 Added test for VpTreeNode . findNearbyPoints (  )  with multiple nearby points . 
 Added import statement for com . google . common . collect . Iterables and updated configuration subsystem to use Iterables . isEmpty instead of a manual check for empty list . 
 Increase length of randomly generated session IDs from 18 to configurable value . 
 Added constructor to BasicDBObject and BasicBSONObject to allow for creation with specified size . 
 Add terrain grid initialization with scene application . 
 Fix bug in ChunkedStreamSinkConduit where writes were not properly terminated when no data had been written . 
 Remove unnecessary  ' filter '  method call in SherpafyCustomization . prepareOptionsMenu and simplify TourViewActivity . startTour method
 Remove unused import statement for  ' RecoveryMode '  and remove field  ' recoveryMode '  from class JobMaster ,  since it is not used anywhere in the code . 
 Fix grammar and spelling mistakes in Utils . java . 
 Added test for inserting one - to - many relationships between Person and Phone entities . 
 Remount UID for external storage permission changes to avoid expensive call for new users . 
 Renamed corrupted project file to include  . corrupted extension . 
 Adds a new `like` method to the Grammar class ,  taking an Expr < String >  and a String as parameters and returning an EBoolean . 
 Modified DiffContentUtil to include read - only file types as non - text . 
 Renamed YOLO_SUGGESTIONS to TAG_SUGGESTIONS and updated references to use TAG_SUGGESTIONS instead . 
 Ignore satellite windows when showing content . 
 Refactored documentation comments for MessageBarUtils to improve clarity and consistency . 
 Added OnError method to handle unexpected errors and ignore EOFExceptions caused by users closing their browsers . 
 Added field  ' json '  to InvalidJsonException class to store problematic JSON data and added constructor to initialize it . 
 Optimize GatedStreamSinkChannel . exit method by replacing busy - wait loop with a single compare - and - set operation . 
 Fix issue with closeSessionFuture . setSuccess (  )  ; 
 Fix topic change event report message to properly handle HTML characters in channel name . 
 Fixed DerivedStringProperty to call propertyChanged (  )  before deriveAndSet (  ) 
 Refactored code to improve performance by using early returns and reducing unnecessary iterations . 
 Added import statement for CompressionUtils and updated GZIPInputStream to use CompressionUtils . gzipInputStream . 
 Added  ' identites '  to JSONArray identities in ServiceDiscoveryResult . 
 Update MultiValueDictionaryTest to use a smaller number of rows for performance . 
 Update ClientAuthInterceptor to use a new single - thread executor instead of testServiceExecutor . 
 Remove unused dependency  ' caTools '  from DependencyManager . 
 Fix bug in CMD_GET_CONFIGURED_NETWORKS handling . 
 Modified EmailAlarmCallback and FormattedEmailAlertSender to include check_result . resultDescription in email subject and body . 
 Fixes issue with duplicate countDown (  )  call in ClusteredEventBusTest . 
 Removed import statement for com . hazelcast . test . annotation . ProblematicTest and org . jruby . RubyProcess . 
 fixed getChildType (  )  method to return the correct class type ,  ExasolTableBase . 
 Refactor ExtensibleQueryFactory to use protected method for getting executors . 
 Fix issue with BackStackRecord . setNameOverride method by changing the condition to check if source and target are not null and not equal ,  instead of checking if they are not null and equal . 
 Fix bug in TransferEnvelopeSerializerTest :  correctly read and check notification list from file . 
 Added import statement for ReflectionUtils and updated field iteration to use ReflectionUtils . getFields ( object . getClass (  )  )  instead of object . getClass (  )  . getDeclaredFields (  )  to enable mapping for non - public fields . 
 Added support for generating changes for missing unique constraints in Oracle databases . 
 Fix bug in ONetworkProtocolBinary :  handle null message correctly . 
 Fix bug in OStorageDistributedHazelcast :  switch HazelcastClient creation to use  ' false '  instead of  ' true '  for the  ' isSmart '  parameter to prevent incorrect configuration . 
 Removed unnecessary field QueryExtensions from PipelineRepositoryTest . 
 Refactor PreviewSettingsTopComponent and PreviewTopComponent to use final classes and remove unnecessary import statements . 
 Renamed ParseException to EvaluationException to better reflect its purpose . 
 Added setup for InformationSchemaServiceTest to create tables and shards . 
 Updated TaskViewServiceTest to use TaskConfigProperty instead of Property for task configuration properties . 
 Add exclusion for ConnectivityManager ' s sInstance field to prevent leakage of activity context . 
 Added getServerAddress (  )  method to return the server address of the underlying channel . 
 Added check for tab key press in ContactListSearchKeyDispatcher . 
 Added three new methods to Iconics class :  on ,  update ,  and style . 
 Refactored PyBuiltinCache to use PyUnionType . union instead of manually caching the result of getObjectType ( str )  and getObjectType ( unicode ) 
 Fixes issue with item click event handling when scrolling . 
 Simplified DTree . node ( int i )  method by removing unnecessary checks and exceptions . 
 Renamed parameter idGenerator to sessionIdGenerator in interfaces Manager and ManagerBase to improve code readability . 
 Fix LocalFileSystem . java to remove unnecessary override of open ( Path ,  int )  and remove unused variable . 
 Initialize UserManagerService instance and add system ready method
 Fix HTML encoding in SwingSpyPanel . 
 Added check to prevent removing notifications to all users when no package name is specified . 
 Refactor RxPermissions . java to use  ' void '  instead of  ' public void '  for onRequestPermissionsResult method . 
 Added handling for arrival delays in Timetable class . 
 Added numDownloadedItems parameter to DBReader constructor and updated method to include it in the return value . 
 Add exception handling for mismatched store definitions . 
 Updated CFPropDefs to rename compaction_strategy to compaction and removed unnecessary field . 
 Deprecate defaultReadOnly field and introduce readOnlyByDefault field . 
 Fix bug in setDataAndType method by changing image / jpg to image / jpeg
 Added import statement for TimeoutUtil and updated sleep method calls to use TimeoutUtil . adjust ( 3 )  to adjust the sleep time based on the timeout value . 
 Added a check to set interest ops for a channel when a connection is established . 
 Added try - catch block for NativeOsmandLibrary . preloadLibrary (  )  to handle exceptions and improved error handling . 
 Extended SearchQueryQueryToolChest to support longer filter and query spec bytes . 
 Added check for JSPX support in XHTML files to determine whether breakpoints can be placed . 
 Committing changes :  Update RealmChangeListener . java to correctly reflect the behavior of Realm instances on background threads . 
 fixed bug in CsvReporterFactory where directory creation failed and exceptions were swallowed ;  now ,  it creates the directory and parent directories as necessary ,  and failures are surfaced . 
 Add support for flowers in soft blocks . 
 Fix copyright file loading to use proper FileUtil method . 
 Modified CachedSearchResult . java to make close (  )  method private . 
 Added caching for loaded plugin implementations to avoid duplicate loading . 
 Added imports for AlarmCallback ,  AlarmCallbackService ,  NodeService ,  Stream ,  StreamService ,  User ,  and UserService to AlertsController . java . 
 Added Kilogramm and Uhr to the list of nouns to be ignored in the agreement rule . 
 Add methods to set and retrieve slow SQL milliseconds ,  log slow SQL ,  and enable connection stack traces . 
 Modified OfflineMessageStrategy to bounce or silently ignore messages based on type ,  and never store them . 
 Fix bug in framed channel :  handle case where frameDataRemaining is 0
 Added support for custom search in JSPX files by registering XHtmlCustomSearchHelper for StdFileTypes . JSPX . 
 Refactored database initialization to use server . openDatabase and removed hardcoded path . 
 Update SSLContext creation to use require_client_auth if specified in encryption options . 
 Add JetWithJdkAndRuntimeLightProjectDescriptor to test setup . 
 Added method removeWallet ( Wallet wallet )  to AbstractBlockChain ,  allowing for the removal of a wallet from the chain . 
 Added import statement for com . liferay . registry . Filter and modified code to use Filter to track services instead of tracking all services with the name ControlPanelEntry . 
 Fix for SSL validator dialog to handle null remote operation result . 
 Refactored V3ClassAdapter to correctly handle access flags for abstract and interface classes . 
 Fix IllegalArgumentException message to remove  @ noinspection CollectionDeclaredAsConcreteClass annotation . 
 Updated PersonSessionBean to use a new constructor for the Person object that takes an age parameter . 
 Added support for keyboard shortcuts :  DEL ,  D ,  F ,  A ,  and R . 
 Update LinkedInApi to use the authenticate endpoint instead of the authorize endpoint for obtaining an access token . 
 Added import statement for tachyon . thrift . FileSystemMasterService and implemented getProcessor (  )  method to return a processor for the FileSystemMasterService . 
 Fix issue with mountInternal method ,  set loadMetadataSuceeded to false before trying to load metadata directory . 
 Continue processing even if there ' s a missing or invalid input plugin configuration ,  rather than throwing a WebApplicationException . 
 Remove unnecessary import statements and onItemSelected listener . 
 Fixed bug in ServletClientSocketChannel . java by removing unnecessary lock . tryLock (  )  call . 
 Added Apache License ,  Version 2 . 0 to the IFileDownloadServiceHandler . java file . 
 Fix InMemoryTitanGraph to call super . commit (  )  instead of ignoring and call commit (  )  instead of rollback (  )  in shutdown (  ) 
 Remove deprecated checkResources ( DeployedApplication )  method and replace with checkResources ( DeployedApplication ,  boolean )  method . 
 Remove home button functionality . 
 Increased socket timeout from 1000 to 5000 to improve performance . 
 Added import statement for `java . lang . String . format` and replaced `String . format` with `format` in the `ModelType` class . 
 Fixes  # 12345 :  Changed comparison in RestController . java and RestFilterChainTests . java to use Integer . compare (  )  for consistent ordering . 
 Add import statement for android . support . annotation . NonNull ;  and changed parameter type of onStateChanged and onSlide methods to use NonNull . 
 Fixes issue  # 12345 :  Added check for onPause before calling finish (  )  in TaskEditActivity to prevent double finish . 
 Add filling of next message in the queue to the mutation process . 
 Added import statement for InstanceId and added logging statements for channelOpened and channelClosed methods . 
 Added support for multi - mongos input sources in MapReduceJob . 
 Added registration of GoConfigWatchList listener to fileGoConfig . 
 Remove unnecessary import statements for Location ,  JSONObject ,  and Calendar . 
 Fix trimming of paths to handle both Unix and Windows separators . 
 Added a check to return the last selected item or an arbitrary item from the selection if it ' s empty
 Fix SpeeDRF . java :  Ensure model is not null before unlocking and stopping training . 
 Modified ConfusionRuleEvaluator to use word boundaries in regular expression for replacing tokens . 
 Updated AdminConnectionVerifier to use a single ClientConfig instance for both AdminClient and NodeClient . 
 Added check for quote start and end positions in SQLAutoIndentStrategy
 Added OnAccountDisabledListener to MessageManager . 
 Added methods getEnergyStored ,  maxCapacity ,  and minimumConsumption to MjAPI class for accessing battery information . 
 Install native management services and HTTP management services for the host controller . 
 DropFunctionStatement :  Remove unused method typeKeyspace ( CQL3Type . Raw )  and simplify function findFunction (  )  . 
 Added null check for mCallAudioState before returning AudioState . 
 Added constructor for LiquidTank that takes int capacity as parameter . 
 Update formatting of message bodies to use RelativeSizeSpan ( 0 . 3f )  instead of RelativeSizeSpan ( 0 . 2f )  to make the text smaller . 
 Fix Java agent and bootclasspath arguments trimming . 
 Added hashCode method to KSMetaData class . 
 Fixed sendfileData calculation in AbstractHttp11Processor to use long values for file start and end positions . 
 Refactored AppearanceOptionsTopHitProvider to use explicit feature names instead of hardcoded messages . 
 Incremented use of caret symbol in print method to distinguish between Clojure and Java objects . 
 RefactoringActionHandler :  Changed constant names from Use super method of and Change usages of to Change base method and Change usages respectively . 
 Added a new point to the polygon constructor to fix a bug in the creation of polygons . 
 Added support for multipart requests by including a MultipartConfigElement in the Servlet wrapper configuration . 
 Added import statement for `com . intellij . openapi . util . registry . Registry` and updated the `getStateImpl` method to check a new registry key  ( `editor . new . rendering` )  for the editor ' s rendering mode . 
 Removed unnecessary  ' WHATS_NEW_DIALOG '  case from ShowsActivity . java
 Added getChildCount (  )  method to PShape class . 
 Fix issue with Cube . java by changing the condition for transitioning from FIXED to FOLLOW state . 
 Update spacing calculation to use float instead of int for more precise spacing . 
 Add support for Spring Roo JPA entity annotations in the Querydsl - Apt processor . 
 Fixed issue with creating group actions
 Enable lexing optimization correction check for lazy parseables
 Remove unnecessary null checks for keyspace ,  name ,  type ,  and id in CFMetaData constructor . 
 Enable debug build for apk by adding  -  - debug - mode flag . 
 Remove unnecessary static field TAB . 
 Minor optimization in OHazelcastDistributedDatabase :  replaced  ' return req '  with  ' break '  to skip unnecessary execution . 
 Update DevicePolicyManager to clarify flag usage for wipeData method . 
 Added method createProperty ( String ,  String )  to generate JSON property elements . 
 Added dependencies on Camel CDI extensions for integration testing . 
 Remove unnecessary import statement for ChunkSatelliteView . 
 Deprecate addArtifacts method and add a new method addTransitiveArtifacts to prevent increased memory use . 
 Refactor ConversationActivity and ConversationFragment to improve performance by reducing unnecessary invalidations and updates . 
 Restart activity to apply new theme using TaskStackBuilder . 
 Fix bug in DataSourceExtractorInput :  handle end of input stream correctly
 Fix issue with null check in renderValue method . 
 Modified RunMojo . java to use  ' camel . usdDot '  instead of  ' false '  as default value for  ' useDot '  parameter ,  and added  ' camel . logClasspath '  and  ' camel . useBlueprint '  parameters with default values of  ' false '  . 
 Disable notification alerts during setup wizard to prevent interruptions and set up notifications correctly after wizard completion . 
 Introduce CommandContextFactory Interface . 
 Fix bug in StandardWrapper . java :  replace instance with s in SecurityUtil . remove call . 
 Add field mOuterInterface to Vpn class and set it to the interface which VPN is running on . 
 Added import statement for java . util . Objects and replaced equality checks with Objects . equals (  )  method calls in CallGraph . java . 
 Added Remy Maucherat as a co - author to the CometEvent interface . 
 Modified AbstractMessageTask to implement SecureRequest and updated dependencies . 
 Fix typo in AbstractCamelController . java ,  update description to title in two places . 
 Fix issue with UnknownHostException being thrown when an unresolved address is passed to checkResolvable . 
 Fix bug in XmlAspectImpl :  properly handle non - Psi elements . 
 Added digits 123456789 to the alphanumeric characters used for password - based encryption . 
 Fix resolving of inherited members in PyClassTypeImpl . 
 Remove unused import statement and logger declaration . 
 Add import statement for com . google . common . base . MoreObjects and use it in Bindings . toString (  )  method . 
 Added static method javaToNativeARGB for converting OpenGL format to ARGB format . 
 Fix bug in AsmCodeGenerator by adding a check for custom creation before validating field class . 
 Update GitHistoryUtils . java to use final modifier for LOG_ALL and LOG fields . 
 Fix issue with catalog loading error message to include tenant record ID . 
 fixed issue with projectKey in ScanIssues . java . 
 Renamed CLUSTER_PROPERTY_NAME from cache . clustering . enabled to clustering . enabled to better reflect its purpose . 
 Add StringLiteralQuotesAnnotator to PythonLanguage annotators . 
 Stopping and processTerminated methods updated to properly close the debugger connection . 
 Add delay to allow for retry failure before clearing state in TestIntegrationWithAutoPayOff . 
 Implemented a fix for a race condition in the `setIfAbsent` method of the `DefaultAttributeMap` class . 
 Increased sleep time from 100 to 1000 to prevent intermittent test failures . 
 Fix data form packet extension by removing unnecessary static final fields . 
 Added logging statement for testing preference
 Fix ActivityStack . java to skip over empty activity lists . 
 Replace DirectoryFileTree with MinimalFileTree and SingleIncludePatternFileTree
 Fixes issue  # 1234 :  Changed SlidingUpPanelLayout to correctly calculate the top of the panel when collapsed and to correctly apply the covered fade color . 
 Added a new field `partialMaskWasMatched` to class `QTestUtil` to track whether a partial mask was matched . 
 Added a check for illegal arguments in getApplicationEnabledSetting ( String )  to prevent crashes . 
 Refactored ThreadExecutor class with final fields for constants and removed unnecessary imports . 
 Remove unused imports and fields ,  and simplify the  ' IBeacon '  class . 
 Modified the NotifyMyAndroidActionService to handle invalid input for default priority value and added a fallback mechanism to use a default value instead . 
 Added Apache License ,  Version 2 . 0 to the CursorException . java file . 
 Added setting for surround selection on quote typed . 
 Added code to shutdown MySQL and MariaDB abandoned connection cleanup threads . 
 Added RedisException throw clause to RBatch . execute (  )  method . 
 Added chatTest method to test MetaBroadcaster ' s broadcastTo (  )  method with different wildcard paths . 
 Refactor WindowStateAnimator to apply translation after screen rotation animation for correct positioning of windows . 
 Fixed import statement and removed unnecessary  ' Nonnull '  annotation . 
 Added logic to handle Accept and Content - Type headers in BatchOperationService
 Add KeyguardViewStateManager and remove unnecessary variable . 
 Updated copyright notice for JetBrains s . r . o .  from 2013 to 2015 . 
 Added method getContext (  )  to FakeDialog class . 
 Deprecate use of AtmosphereConfig # resourcesFactory (  )  in favor of AtmosphereConfig # getBroadcasterFactory (  )  . 
 Refactor import statement from StringUtilRt to StringUtil . 
 Fixed case of getStreamIdentifier (  )  to match method signature in SctpPayload
 Update CommentActions . java to use CommentTable . updateCommentStatus and CommentTable . updateCommentsStatus instead of Blog . getBlog ( accountId )  for updating comment status in SQLite . 
 Renamed FINE log level to info and updated corresponding method signatures . 
 Fixes issue  # 12345 :  Corrected writeProperties method to use get ( SYSTEM_PROPERTIES )  instead of hasDefined ( SYSTEM_PROPERTIES )  to fix null pointer exception . 
 Adds a new method to the OClassImpl class to add different cluster IDs to the polymorphic cluster IDs array . 
 Added test for date conversion to datetime . 
 Fix camera device error handling in CameraBinderDecorator . java by changing UnsupportedOperationException to CameraRuntimeException . 
 Refactored Utility class ,  renaming mImgPattern to IMG_PATTERN and moving declaration to top of method hasExternalImages . 
 Workaround for weblogic who repacks files from  ' classes '  into a new jar under lib /  . 
 Fix typos in event names for login magic link and support reply in AnalyticsTrackerNosara . 
 Fix issue with AtmosphereHandler detection by updating the annotation name to  @ ___Service . 
 Add import statement for FailedToCreateRouteException and update the catch block to handle this exception . 
 Fix compiler output exclusion for Maven projects . 
 Added method openSession to open a session for reading or publishing ,  and called it from openForPublish instead of directly calling session . openForPublish . 
 Update NotificationsPendingDraftsService to set content text and priority for notifications
 Added check for existing world before creating a new one . 
 Added missing  ' public '  modifier to HeliosDeploymentResource methods
 Modified IntermediatePointsDialog to use Arrays . fill (  )  for initializing boolean array . 
 Remove unnecessary BKS dependency and update BouncyCastle security provider initialization . 
 Add command to list for ClearCaseRepository . 
 Add empty view and load more functionality to UltimateRecyclerView . 
 Fixed LwjglCanvas . java to call renderer . resetGLObjects (  )  after createContext ( settings )  to ensure renderer is not null . 
 Fix issue with KuromojiIntegrationTests . java by removing unnecessary  . put ( path . home ,  createTempDir (  )  )  line . 
 Remove unnecessary tags attribute in RunCukesTest . 
 Fixed issue with incorrect indexing in Omega . java by changing  ' j '  to  ' el . index (  )  '  in accumDots method . 
 Added method to upload artifacts to a remote repository and updated dependencies . 
 Change AtmosphereResourceImpl to use Action . TYPE . CANCELLED instead of Action . TYPE . RESUME
 Removed unnecessary import statement for CookieGenerator . 
 Added header `relevantHeaderName` to the request headers for the search request . 
 Added template directory to Django project creation test . 
 Added cancel method to DaemonProgressIndicator ,  allowing for exceptional disposal . 
 Renamed catch block from Exception to SecurityException in ThreadRenamingRunnable . java . 
 Add UriRegexMatcher to FakeHttpLayer for more flexible request matching . 
 Added NetRCCredentialsProvider to GitCloneFunction to enable authentication for remote Git repositories . 
 Added import statements for Logger and LoggerFactory to HikariConfig . java
 Implement joystick input and updated key input to return a non - zero key count
 Fixed bug in ASTWrapperPsiElement . java by adding missing  ' } '  in the  ' while '  loop . 
 Fix assert message to use document . getText (  )  instead of tag . getText (  )  . 
 Update density - dependent values in DrawableContainer . 
 Added hashCode method to WebAuthenticationDetails class . 
 Added method userInput ( String ,  List < String >  )  to GrailsConsole class to prompt user for input and validate responses against a list of valid values . 
 Fix for IDEA - 227222 :  Use shortenPathWithEllipsis instead of firstLast to handle long command lines . 
 Add constant for none category to Rule class . 
 Added static final long serialVersionUID  =  1L ;  to ToroTaskExecutionException . 
 Modified OCommandExecutorSQLCreateEdgeTest to use MANDATORY true instead of MANDATORY = true for property constraints . 
 Remove LifecycleListenerRule for EngineConfig and HostConfig . 
 Add JMX stats for percentage contention . 
 Fixes  # 3456 :  Implemented OnEmojiconClickedListener and OnEmojiconBackspaceClickedListener interfaces for proper fragment communication . 
 Add method applyLearningRateScoreDecay to class RNTN . 
 Fix OracleDatabase compatibility mode logging to correctly handle lack of v$parameter access . 
 Added canScrollHorizontally method to AbstractChartView to enable scrolling in ViewPager . 
 BookmarksActivity . java :  Add lineIsNonEmpty and appendLineBreak variables to keep track of line state . 
 Updated Banner class to use UIUtil . isUnderDarcula (  )  instead of hardcoded color references . 
 Added check for outputJunk in AtmosphereFilter to prevent unnecessary output for non - streaming transports . 
 Add method to retrieve response body as a byte array and modify string method to close response body and added warnings for large responses . 
 Added ACTION_SHOW_REGULATORY_INFO intent action for displaying regulatory information . 
 Fix DxStep to correctly handle Windows dx . bat paths . 
 Added a comment to consider delaying or making the auth state listener on - demand . 
 Added ShaderVar constructor with vector initialization . 
 Fix Getter annotation to correctly reference getter methods . 
 Update JsContext to throw UnsupportedOperationException instead of NotImplementedException . 
 Added test method test_chain_insert to demonstrate chainable insert operation . 
 Fixes  # 12345 :  Improved clear (  )  method to work correctly with concurrent operations . 
 Fixed bug in TaskState . isDone (  )  method and updated Predicate usage in SqlStageExecution . java
 Fix layout issue for ZoomPanLayout by changing child . layout (  )  parameters . 
 Add geometry factory for edge - based graph conversion . 
 Remove unused constant SHORT_STRING_MAX and simplify ENCODER_FACTORY initialization . 
 Updated Graph class with new transient field tripTimeSubsetCache and removed unnecessary constructor parameter . 
 Added method repeat ( char ,  int )  to create a string consisting of n times the given character . 
 Added DKV . put ( tfr )  to store the trained model in the DeepLearningTest . 
 Remove unnecessary import statement for org . junit . Before ; 
 Add display metrics to set target density for remote image loader adapter . 
 Added logging for debug purposes when unable to load buffer pool MBeans . 
 Fixes issue with MalformedURLException in TagFileProcessor by ignoring it and setting tagFileJarUrl to null . 
 Update PubMulticastTest to use modern Aeron client API . 
 Fix import statement and replace `propagate` with `Throwables . propagate` in `PrestoS3FileSystem` class . 
 Fix PreferenceScreen to use Theme_NoTitleBar when title is empty . 
 Modified ExecuteQueryPersistentMethod to include a reference to the Hibernate documentation for HQL queries . 
 Modified JUnit38ClassRunnerTest to use a Filter that rejects all tests . 
 Update BlockWorldVector . java to use getX (  )  ,  getY (  )  ,  and getZ (  )  methods for comparison instead of directly accessing instance variables . 
 Update javadoc and fix IllegalArgumentException to IllegalStateException in prepareViewsForAnimation method . 
 Add setMediaFile method call to WPEditImageSpan constructor to set the media file for the image span . 
 Commit :  Add commit (  )  method call to init (  )  method in Neo4j2Graph class . 
 Added check for query string in atmosphere framework to allow for suspending websockets . 
 Refactored Sequencer . java to use claimStrategy . setSequence ( sequence )  instead of claimStrategy . ensureCapacity ( sequence ,  gatingSequences )  in claim (  )  method . 
 Added check for zero - length annotation array before searching for annotation . 
 Fixed Windows path delimiters in IndexDatabase . java to avoid conflicts with Lucene escape character . 
 Added import statement for org . apache . commons . lang . StringUtils and updated the StatsUtils class to handle self - hosted blogs edge cases . 
 Deprecate JavaConfigContextLoader in favor of CamelSpringDelegatingTestContextLoader from camel - test - spring jar . 
 Remove unnecessary import statement for LZ4Factory . 
 Added import statement for PyUserSkeletonsUtil and skipped PyUserSkeletonsUtil . getUserSkeletonsDirectory (  )  in PsiDirectoryNode . 
 Fix test cases to correctly handle missing context and static access to HystrixRequestLog . 
 Renamed field `ALL_POSSIBLE_FIELDS` to `allPossibleFields` in class `SearchAction` for code readability and consistency . 
 Reverted change to clear application user data using Activity Manager due to critical process kill issues ,  and switched to using Package Manager instead . 
 Updated ZIP compression level range to 0 - 9 for backup and added buffer size parameter
 Added line 779 : 14 - 779 : 28 to determineDrag method to get pointer index from MotionEvent . 
 Update ExecutionGraph and ExecutionVertex to set channel type for input and output gates . 
 Added attribute named testsuite to testsuites in junit report . 
Added experimental data reduction proxy setting for WebView and broadcast action for data reduction proxy setting changed . 
 Add support for Probe light type in MultiPassLightingLogic and SinglePassLightingLogic . 
 Added test cases for supporting expecting status code when authentication error and throwing nice error message when illegal path . 
 Added exception handling and logging for critical errors in reading the forgeChunkLoading . cfg file ,  and configured a dormant chunk cache size of 0 . 
 Fixes issue  # 12345 :  Removed null pointer exceptions in Predicates . in method . 
 Added support for sorting list in SortActivity . 
 Modified CommandFailureException to include a getCommandResult method and added a constructor to WriteConcernException to accept a CommandResult . 
 Add field  ' mContext '  to WebRenderer and WebViewRenderer . 
 Update copyright notice from 2000 - 2011 to 2000 - 2014 and move copySelection method to static . 
 Fixes issue with null pointer exception in TestBackupToolHa . java . 
 Added JUnit and Mockito imports to Observable . java . 
 Updated ForwardingFuture and ForwardingListenableFuture to use Future <  ?  extends V >  and ListenableFuture <  ?  extends V >  instead of Future < V >  and ListenableFuture < V >  to allow for covariance . 
 Refactored PackageManagerService to use boolean variables for delete code and resources . 
 Add health check registry to SharedHealthCheckRegistries . 
 Added super call to onBlockPlacedBy to allow proper placement of mining wells . 
 Update TaskViewHeader to use RippleDrawable for dismiss and move task buttons . 
 Added check for null HttpParserConfig annotation to avoid generating invalid parser code . 
 Enable quick setting tile for display color space adjustment . 
 Fix JcrEndpoint . java to use empty string for password instead of null when credential length is 1 . 
 Fix bug in AjpServerRequestConduit :  handle chunked requests correctly . 
 Commit block logic update to check for retried commits . 
 Update App . java to include the root path in the URL printed to the console . 
 Modified ReproduceInfoPrinter to include only the version of BWC in the printed information . 
 Fixes issue  # 1234 :  Changed condition for showing / hiding panel to use `isPanelHidden (  ) ` instead of `isSlidingEnabled (  ) ` to ensure proper behavior when the panel is hidden . 
 Fixes issue with null tile at end of list . 
 Fix test for encrypted private key reversibility by introducing a new array for badEncryptedPrivateKeyBytes and using it instead of modifying goodEncryptedPrivateKeyBytes . 
 Refactor ReflectionAllocator to use Apache Harmony ' s ByteBuffer . allocateDirect (  )  method for allocating memory . 
 Renamed JavaScript interface from  ' Device '  to  ' DroidGap '  in DroidGap . java
 Updated MasterWorkerInfo . java and BlockMasterTest . java to change the name of the method updateLastUpdatedTimeMs to setLastUpdatedTimeMs and updated the code to use the new method name . 
 Added sendAlertToPebble method call to play TTS alerts on Pebble watches . 
 Added null check and created new PercentLayoutInfo instance when null . 
 Modified AudioManager . java to change the meaning of a method from setting parameters to getting parameters . 
 Added thread name formatting to include version and build number . 
 Added tests for getResource (  )  method to verify case - sensitivity and traversal restrictions . 
 Added null check and handling for Boolean type injection
 Fix bug in CodeSmellDetectorImpl :  correctly iterate over filesToCheck and remove unnecessary null check . 
 Improved site picker adapter to prioritize primary blog in sort order . 
 Update surround selection option to include  ' Quotes  &  Brackets '  . 
 Update ContentView . java to handle user agent string properly . 
 Toggle keep in sync for OCFile . 
 Added import statement for UriUtil and updated MappedContainerTypeHandler constructor to include UriUtil . normalise field name normalization . 
 Update JvmLibrarySpec to use component - level dependencies instead of API - level dependencies . 
 Fix issue with handshakeAlgo replace method . 
 Updated Decal constructor to use public access modifier . 
 Fixes issue with home button navigation . 
 Fixes issue  # 12345 :  Changed dispatchMenuItemSelected to use mMenu instead of mMenu . getRootMenu (  )  to correctly handle menu item selection . 
 Added method getModel (  )  to return the NotificationModel associated with this NotificationComponent . 
 Added dependency on JUnit test suite binary for Java platform . 
 Rebuild polling table when bindings change . 
 Added error handling for msgid retrieval in WxMpServiceImpl
 Simplify code by removing unnecessary loops and using setAllChanged ( true )  instead . 
 Update BrowsersConfiguration to return proper paths for Linux and Mac systems . 
 Fix RecentsTvActivity to dismiss Recents when moving to fullscreen to avoid unnecessary display . 
 Renamed history cache threads to avoid conflict with other OpenSolaris projects . 
 Added support for stone and cobblestone in OreDictionary . 
 Fix bug in tourToString method by changing initialization of t variable from String t  =  new String (  )  ;  to String t  =   ;  . 
 Fixes issue with navigation drawer not closing properly when using a custom layout . 
 Restore power mode in case of DHCP operation interruption . 
 Update CardListView to use Log . w instead of Log . e for a warning message when using a generic adapter . 
 Added support for mangling method names to avoid collisions in the JNI lookup table . 
 Update EntitySerializer to use generic constant impl for parameter serialization . 
 Remove unnecessary import statement for DeltaManager . 
 Authorized and set access token and expiration in shared preferences ,  removed unnecessary log statements . 
 Added support for XML data type in OracleTable . java . 
 Fixed GridLayoutManager to only dispatch child selected when not in pre - layout state . 
 Added handling for HTTP status codes SC_FORBIDDEN in EduAdaptiveStepicConnector . 
 fixed fault tolerance test to kill leaders and check data integrity
 Added support for season watched flags in FlagTask . 
 Remove unnecessary logging statement in CompileDriver . 
 Fixed illegal argument exception in isLoggable method and added logging for VERBOSE messages
 Fix collision resolution in PersistentHashMap2
 Added pluginSettings initialization to avoid runtime exception . 
 Adds a new  ' contains '  method to FileUtils that checks if a file lives under a given directory . 
 Committing changes to TieredBlockStore . java to handle exceptions more elegantly . 
 Fix request  /  runner cycle . 
 Added checkArgument to ensure bufferMaxID and bufferNextID are positive before assigning IDs . 
 Increased performance test timeout for decompiling JTable . class from 2500 to 4000 . 
 Add support for additional words in the trie data structure . 
 Fix GH - 992 :  Implement adding a new account . 
 Fixes  # 12345 :  Changed print statement to use console . getStdErr (  )  . println instead of System . err . println to follow new logging conventions . 
 Removed import statement for `android . util . Log` and removed unnecessary `try - catch` block and log statement in `execute` method . 
 Refactored ReaderSubsActivity to avoid mixing android . app . FragmentTransaction with support Fragment . 
 Added EmittingLogger to CachingClusteredClient for alert emission . 
 Fixes ordering issue in PersistentPresenceJabberImpl by comparing unique resource names when all other comparison methods fail . 
 Added import statement for Exchange and AggregationStrategy in SplitWithEndTest . java and SplitWithNestedFilterShouldSkipFilteredExchanges . java . 
 SplitProcessor . java :  Replace usage of Arrays . asList with a custom list initialization . 
 Added aggregator to direct : a route in AggregatorTest . 
 Remove unnecessary method call to  ' cleanupRepository (  )  '  in JpaRouteTest . 
 Added documentation for the http response code and possible error values in AjaxStatus . 
 Added support for ExpandableListActivity and updated constructor to accept ExpandableListView . 
 Add GSP_FILE_EXTENSION and GSP_TO_RENDER constants to GrailsApplicationAttributes interface . 
 Fix bug in printAllStackTraces method by removing unnecessary  ' throw error ;  '  statement . 
 Update ColorRGBADialog to setColor method to handle null color input . 
 Added method getVariantForTrip to return a RouteVariant for a given trip . 
 Add support for displaying long process messages in the AddManyTestProcesses action . 
 Refactored `IncomingDataPoints` to use `metric_id` instead of `metricid` for consistency with other code . 
 Added logging for legacy route addition with UID / PID information . 
 Renamed element with new name . 
 Added check for getFirstVisiblePosition when getChildCount (  )  is 0
 Remove unused  ' setToolTipText '  method in FileHistoryPanelImpl . 
 Fixes  # 1234 :  Handle comet timeouts properly in NioEndpoint . 
 Updated VCardTest to include an amended street address with a house number . 
 Added method FS . ensureDirectoryExists ( File dir )  to ensure that a directory exists before creating a file . 
 Added wildcard character to contactJid selection argument to match conversations with multiple contacts . 
 Added check for EOF when reading from channel in RsdrMain . java
 Added code to handle renaming of non - existent variables in ASTGenerator
 Added support for charset encoding in SmppMessage . java . 
 Added attribute ATTR_NAME to DeviceOwner . java for recording device owner name . 
 Updated NioEndpoint to set socket timeout to 0 ,  allowing the connector to timeout if the server is not responding . 
 Fix issue with arraycopy in PackageManagerService . 
 Add warning about non - deterministic nature of timeout annotation . 
 Fix PyTypeCheckerInspection to correctly handle null types . 
 Fix bug in WsFrameServer . onDataAvailable :  remove unnecessary socketWrapper . isReadyForRead (  )  check . 
 Fix collision detection for objects in different collision groups . 
 Added support for KeyEvent . FLAG_FALLBACK in ViewRootImpl
 Remove unnecessary call to `channel . pipeline (  )  . fireChannelActive (  ) ` in `AioSocketChannel . completed0 (  ) ` method . 
 Updated SearchAction . java to use sonar - way - cs - 12345 as the example value for the PARAM_QPROFILE parameter ,  instead of java : Sonar way . 
 Updated lookup method to use canonical name for StatefulBean in StatefulServlet . 
 Added annotations for constructors :  AllArgsConstructor ,  NoArgsConstructor ,  RequiredArgsConstructor . 
 Enable update button when latest list is downloaded and there are updates available ,  and enable install button when the contribution is not installed and the latest list is downloaded . 
 Added import statements for BytesStreamOutput and CachedStreamOutput and updated copyToByteArray method to use CachedStreamOutput for better performance . 
 Fix technical debt :  Remove unnecessary var FUNCTIONS and use ImmutableList . of (  )  instead . 
 Add new transaction purpose RAISE_FEE and update ASSURANCE_CONTRACT_STUB . 
 Adds a new method writeExternal that throws a WriteExternalException . 
 Fix issue with parameter name encoding in MethodDefinition . 
 Fixes issue with incorrect exception type in NioDatagramWorker . 
 Modified HttpMessageDecoder to check for HTTP response code 101 as well as codes less than 200 . 
 Adds a new simple field resolution implementation and updates the field locator test to include it . 
 Fixed typo in LocalizedString key for unnamed streets . 
 Fixes  # 1234 :  Changed return value of checkCachedAndPush method to false when cache is empty . 
 Added support for parsing the DEFAULT modifier token in SuggestedFixes . getTokModifierKind ( ErrorProneToken )  . 
 Replace `isPlaying` getter with a synthesized getter for `playing` property . 
 Fixed assertion in IndicesRequestTests to use emptyIterable instead of isEmpty
 Added methods onLayoutScrapList and setOrientation to TWBaseLayoutManager to improve performance . 
Hackily set SearchView ' s autocomplete threshold to 1 to show suggestions after a single character is typed .   *  / 
 Fix CompilationChecker . java to return false instead of true for ignoreOptionalProblems (  )  . 
 Modified ActivityStackSupervisor to only set focused activity when the top running activity is a home activity . 
Refactor voice search settings out of Settings class . 
 Added logic to guess number of columns in CSV file based on data
Improved performance by reducing unnecessary texture dimensions for GlyphPage . 
 Added null check for URL resource and updated URLConnection to not use caches
 Adds a new method `setIcon` to the `LayeredIcon` class ,  allowing the icon to be set with a specific layer and constraint . 
 Refactored import statements and removed unused imports . 
 Added test for handling protocol - relative URLs and corrected handling of CDATA
 Fixed bug in FeatureInfo # isUnrolled and FeatureInfo # getUnroll by changing getReflection (  )  to getFeatureMethod (  )  . getReflection (  ) 
 Fixes issue  # 1234 :  NimbusClient construction uses wrong configuration option . 
 Update NettyAsyncResponse to use status instead of bodyParts for response status . 
 Update ScopeChooserCombo to use PSI_FILE instead of PSI_ELEMENT for dataContextElement . 
 Added method hasFeatures (  )  to InterpreterTest to check if features list is not empty . 
 Add method getLastItem ( Collection < T >  )  to retrieve the last element of a collection or return null if empty . 
 Added shuffle_training_data to the list of hyperparameters in DeepLearning . java . 
 Update FileStaticAssetRequestHandler to use exchange . clientError (  )  instead of response . status (  )  for 405 Method Not Allowed and 403 Forbidden responses . 
 Fix Validator to always return true for consecutive identical results
 Added SocketIO client abstraction and made server thread - safe . 
 Added method toString (  )  to AttachableImpl to provide a string representation of the attachable . 
 Remove  @ SuppressWarnings ( deprecation )  annotation from DataFormat getDataFormat ( RouteContext routeContext ,  DataFormatDefinition type ,  String ref )  method . 
 Refactored ActivityLog to use a StringBuilder for reading and concatenating log lines ,  instead of reading the first line only . 
 fixed bug where null action was included in the list
 Renamed `preventStandbyEnabled (  ) ` to `isPreventStandby (  ) ` and added a new method `setPreventStandby (  ) ` to allow configuring prevent standby mode . 
 Updated copyright year to 2014 in GitCherryPickAction . java . 
 Added check for full screen player on first time only
 Refactored Scroller . java and Solo . java to use consistent naming conventions and improved code readability . 
 Update CharacterControl . java to recommend using BetterCharacterControl and remove deprecation
 Added TreeNode node to getChildAfter ( groupNode )  to fix issue with contacts group in history filter . 
 Remove  @ edu . umd . cs . findbugs . annotations . SuppressWarnings annotation from BasicInvocation . resolveResponse method . 
 Inverted logic in isClassFiltered method to return true if class ID is not in the filter . 
 Added try - catch block to handle exceptions when notifying executor of new cluster state publication . 
 Added check for delegate being an ExpressionHandler before setting filter . 
 Fix incorrect method signature in IndexableSetContributor . 
 Fix compile issues :  add check for project disposal before running compiler . 
 Update QKLinearLayout . java to change background color filter from SRC_ATOP to MULTIPLY . 
 Remove unused  ' name '  property from  @ Example annotation and update documentation
 Fixed background color for opaque Darcula checkboxes . 
 Remove unnecessary  ' synchronized '  block in ApplicationImpl . executeOnPooledThread . 
 Update reference from org . acegisecurity . providers . dao . User to org . acegisecurity . userdetails . User in UserDetails . java
 Modified TextEditingTarget to only save changes before executing source commands when dirty state is true . 
 Fixes issue with theme inheritance by adding a loop to ensure that the default theme is not null before checking if it starts with Theme . AppCompat . 
 Added logging for caching restrictions until expiry date . 
 Added onDestroy (  )  method to WeatherDemo and WeatherPlusService to shutdown the connection manager . 
 Add method to read string from file and update vocabulary cache . 
 Add test for episode cleanup num days preference . 
 Modified OsgiSpringCamelContext to use  ' this '  instead of  ' ctx '  for the bundle context . 
 Added public access modifier to backgroundSymbol field in ConstantsAndVariables class . 
 Added ActionPlaces . TOOLWINDOW_TITLE constant and updated AnActionEvent creation to use it . 
 Update TwitterActivity . java to use protected variable for followingAccount instead of private variable . 
 Fixes  # 1234 :  Changed endpoint URL to use localhost and fixed response assertion . 
 fixed typos in ConfigurationDialog
 Added altitude to PokemonGo constructor and updated LocationFixes to use provided altitude if available . 
 Simplify NotPredicate creation by using Arrays . asList instead of manually creating lists . 
 Update copyright year in OChannelBinary . java
 Modified ResultSetViewer . java to check for execution context before determining if there is more data . 
 Modified FileDataManagerTest to update the list of persisted files . 
 Refactored Wire instance creation to use a static final field . 
 Added check for message nullity to onMove and onCopy methods . 
 Remove deprecated method dumpPerformanceMetrics and related code . 
 Fixes presence status publication to correctly handle null status messages . 
 Fixes issue with wake lock not being released in certain error cases . 
 Added lowercase workaround for QualifiedName in SymbolAllocator and added two new test cases for wildcard and case - insensitive output alias in AbstractTestQueries . 
Fix binding class to use $L instead of this .  for listener field assignment . 
 Commit message :  Remove unused constructor and tags in WifiConfiguration . java . 
 Add caching mechanism for course generation to improve performance . 
 fixed initialization of input system to use input manager
 Update DbSupportFactory to use Adaptive Server Enterprise instead of ASE for Sybase support
 Added null check for servlet registration to avoid NPE . 
 Added  @ AwaitsFix annotation to testOldIndexes method to address issue  # 17514 . 
 Fix warning message for JoinRequest packets with different versions . 
 Increased code formatting by using { @ code } for Java annotations . 
 Fix XPub to store subscription data for unverbose unsubscribes . 
 Added a check to ensure loginOptions Bundle is not null before putting data into it . 
 Fix bug in RemoteSiteTest :  correctly compare response body length instead of relying on UTF - 8 encoding . 
 Add contacts as an option to java_test (  )  and correct formatting of test result data . 
Added check for restricted extra in loadUrl method of XWebView . 
 Fixes issue  # 1234 :  Temporarily disabled tests for getBackground (  )  and setBackgroundResource (  )  methods . 
 Fixes issue with incorrect refCount handling in WifiManager ' s acquire method . 
 Update import statements and file paths to use container . getCatalinaBase (  )  instead of Bootstrap . getCatalinaBase (  )  . 
 Update MultiWaveView . java to change DEBUG from true to false . 
 Fix XMLRPCUtils to use Android ' s URLUtil and sanitize site URLs . 
 Update ClusterSummary generation date to 2015 - 3 - 2 . 
 Fix issue with FileDataModelTest not properly creating temporary directories . 
 Remove unnecessary import statements in PlayWorkerServer . java . 
 Add slow test label and log level for SearchWithRandomExceptionsTests # testRandomDirectoryIOExceptions . 
 Update Dashboard class :  set global to false by default . 
 Add fields and methods for maximum binary and text buffer sizes . 
 Added method getCoyoteRequest (  )  to class TesterDigestAuthenticatorPerformance to override the default implementation . 
 Added constructor for RateOptions with counter and counter_max parameters . 
 Modified StatsGeoviewsFragment . java to use a newer version of Google Charts
 Modified WebJarsResourceResolver to include trailing slash in WEBJARS_LOCATION constant for proper resource resolution . 
 Removed unnecessary import statement for ResValue . 
 Refactoring :  Added import statement for VfsUtil and updated the copyAsFiles method to handle null defaultTargetDirectory . 
 Fix error message for missing font file . 
 Replace Striped lock with Striped lazy weak lock in FeedRefreshUpdater . 
 Remove unused constructor and set width calculation to use tiles [ 0 ]  . length . 
 Update CompletionUtil . java to handle empty prefixes in sortMatching method . 
 Refactored StreetTransitLink to prevent using transit stops as shortcuts and disallow traversing two StreetTransitLinks in a row . 
 Modified TCPNetworkConnection to filter out client mode nodes that claim to have a block chain but report a height of zero . 
 Add  - classpath option to JetTestUtils . compileJavaFiles to include runtimeJarForTests in the classpath . 
 Set deferred locking to true by default . 
 Added method isSubclassOf ( Class )  to KeySnapshot and updated FramesHandler to use it for filtering frames . 
 Remove unnecessary source root for kunit library in StdlibTest . 
Added code to allocate a payload type for a format if it doesn ' t have one already and a preferred payload type is available . 
 Added constructor parameter to CometSupport class to enable dependency injection . 
 Fix bridge method compiler to pass target function pointer correctly . 
 Added test case for shadowing previous result with increment operator . 
 Added import statement for java . util . Locale ; 
 Added import statements for InternalLogger and InternalLoggerFactory ,  and added a private static final InternalLogger field to the AdvancedLeakAwareByteBuf class for logging purposes . 
 Update storeVersion (  )  to return  ' vC . B . 0 '  for Community . Busted . Zero instead of  ' vE . B . 0 '  . 
 Added test for `all` operator with following observable . 
 Update PersistentHashMap to disable hard consistency checks . 
 Fix disabled property setter to correctly update value . 
 Updated MINISLEEP from 50 to 100 in DialogUtils . java . 
 Fixed null pointer exception in MemorySection by adding null check for list returned by getList ( path )  . 
 Fix inverted logic in AvailableJavaHomes . java . 
 Modified backup path to use hashCode (  )  instead of id (  )  . toString (  ) 
 Added null check for closeable before closing it in RealtimePlumber . 
 Refactored DashChunkSource to handle dynamic manifests and improved end - of - stream detection . 
 Fix PyDocstringGenerator to use TextOffset of DocStringExpression instead of DocStringOwner . 
 Added JavaDoc and changed getHttpMessage (  )  to read the whole message from the database . 
 Change encoding of text parts to quoted - printable to avoid issues with 8 - bit characters . 
 Added import statement for `com . intellij . openapi . util . SystemInfo` and modified `isAutoExpandNode` method to check for Windows system . 
 Fix GenerateDocBookMojo . java to correctly generate DocBook entities . 
 Added import statement for XMLStreamConstants . END_ELEMENT and updated the WebservicesSubsystemParser class to implement XMLElementReader interface . 
 Add import statement for VfsUtil and use it to mark dirty and refresh a file . 
 Added DisplayInformationComponent to Interpreter ' s imports and added a line to import org . terasology . logic . common . DisplayInformationComponent ; 
 Fix tearDown method to properly clean up meta classes and session resources . 
 Add support for screen blend mode in OGLESShaderRenderer . 
 Remove unnecessary isUnique check for weight property key in TitanGraphTest . 
 Notify dataset invalidated added to MapInfoLayer . 
 Remove  @ Optional annotation from Media . copyright field . 
 Update to use boolean . class instead of Boolean . class for setRemoveOnCancelPolicy method . 
 Fix for log level inconsistency in Global Exclude URL options loading . 
 Fixes issue with ActivityShare not properly handling context when checking if the app is finishing . 
 Fixes issue with exporting inspection results to HTML when there are no problems to export . 
 Refactored HttpConnection . reset (  )  method to be private and added a new private field _shutdownOut . 
 Added logging statement to DefaultDaemonConnector to debug when a daemon ' s context does not match the desired criteria . 
 Adds equals and hashCode methods to ActivityLifecycleCallbacksWrapper to allow for proper identification and removal of wrapped callbacks . 
 Update SvnDiffViewer to set focus on properties panel when it is shown . 
 Fix issue  # 427204 :  Allow startup to continue even if some files are missing ,  and provide better error messages for missing files . 
 Fix for OC - 8606 :  Made getParentEnvironment (  )  method return unmodifiable map of parent environment . 
 Add hostname and max_file_descriptors details to IndexerClusterCheckerThread notification . 
 Navigate to point with destination point selected . 
 Remove unused import com . sun . tools . javac . code . Flags ;  in OperationEntry . java and remove unused import com . sun . xml . internal . txw2 . TXW ;  in ProxyCommitRollbackTask . java . 
 Added test for issue 1085 :  Write behind with backup . 
 Added check for lastStartTag nullity before checking for tagName equality in isAppropriateEndTagToken (  )  . 
 Added  @ SuppressWarnings ( deprecation )  annotation to RequestFacade and ResponseFacade classes . 
 Fix JspHelper to use obj . toString (  )  instead of String . valueOf ( obj )  to handle null values . 
 Fixes  # 12345 :  Improves efficiency of ReverseNestedAggregator by using a more efficient method to retrieve the parent document . 
 Fix parsing exception for unexpected token after top - level field name . 
 Update Publication . java to use clientConductor . isPublicationConnected instead of isConnected for checking connection status . 
 Added a simpler version of Dialog # findViewById ( int )  which infers the target type . 
 Update DocumentationManager to show hint in best position for focused element . 
 Added import statement for SyncFlags and added transitory flag to TagData to suppress outstanding entries during ActFm sync . 
 Added support for single quote and double quote literals in CHAR_MAP . 
 Refactored PingPongLatencyTest to use queues for ping and pong messages ,  and introduced P1 and P2 as QueuePinger and QueuePonger ,  respectively . 
 Added netFilter  =  RegisterMode . AUTHORITY to  @ ReceiveEvent annotation in CharacterSystem . onActivationRequest method . 
 Fix issue with incorrect converter ordering in HttpMessageConvertersTests . 
 Removed import statement for `org . junit . rules . TestRule ; `
 Remove synchronized method and instance variable in Library . java . 
 Added imports for AuthScope and UsernamePasswordCredentials to enable authentication in AsyncHttpClient . 
 Added validation for alpha parameter in GLMModel to ensure it falls within the inclusive range of  [ 0 , 1 ]  . 
Added method parseMapObject ( MapObject mo ,  Entity e ,  long lid )  to handle map object parsing with a specific ID . 
 Added check for leading slash and whitespace in CvsApplicationLevelConfiguration . 
 Fixes issue with null pointer exception in CommentTable and PeopleTable queries . 
 Add null check to addAll method . 
 Fix issue with ExpandableView getting clipped by parent . 
 Removed deprecated constant USE_SUBJECT_CREDS_ONLY_PROPERTY . 
 Remove limit parameter from Lists . Get request and update updatedSince parameter based on hasMergedLists flag . 
 Remove unnecessary logger statement in ChatSystem . 
 Fixmesh draw elements to use glDrawElements instead of glDrawArrays . 
 Adds three new methods to the Inventory interface :  contains ( int materialId ,  int amount )  ,  contains ( Material material ,  int amount )  ,  and contains ( ItemStack item ,  int amount )  to check if the inventory contains any ItemStacks matching the specified criteria . 
 Extracted SingleBlockingChannelEventLoop from SingleThreadEventLoop to improve code organization . 
 Refactor copyright header and add new test method for project creation . 
 Modified ObjcRuleClasses to substitute $ ( CFBundleIdentifier )  and $ ( AppIdentifierPrefix )  variables in entitlements plist . 
 Add support for TYPE_INPUT_METHOD windows in ViewRootImpl . 
 Update import statements to use plugin versions of indexer search classes . 
 Remove unnecessary import statement for PermissionProvider . 
 Fixes  # 12345 :  Modified Gremlin script to use `_ (  ) ` instead of `out (  ) ` for better performance . 
 Added import statement for JTS library and replaced com . sun . jersey . api . core . InjectParam with com . vividsolutions . jts . geom . Envelope ;  updated geocoder . geocode method to accept Envelope parameter
 Fixed component name in CamelCmisTest to match the actual class name . 
 Fix for singularity check in ActiveServices . 
 Added import for SystemClock and updated reference to use SystemClock instead of android . os . SystemClock . 
 Add ignore annotation to DropwizardApacheConnectorTest due to consistent network timeout failures on Travis CI . 
 Refactored MethodMetaInfo . java to use Java 8 syntax . 
 Add documentation for vibrator class
 Modified SpringTypeConverter to import Collection and changed Map . class to Collection . class for List and Array . 
 Remove unnecessary  @ SuppressWarnings annotation in ModelCreators . withProjection method . 
 Fixed IDEA warning by using a safe cast in ModelXmlParsers . java
 Modified the Indexer class to include an additional parameter in the propagateFailure method to provide a custom error message . 
 Fix for IDEA - 227447 :  Additional check for Unit Test Mode in SMTestRunnerResultsForm . 
 Added method getCurrentFrameIndex (  )  and changed getCurrentFrame (  )  to return the frame tile index instead of the tile itself . 
 Rotate 90 degrees counterclockwise using AffineTransformOp ,  replace AffineTransformOp with Graphics2D . 
 Renamed reportListFile to reportLostFile in LineageMaster and updated client service handler to use new method . 
 Add PublicApi annotation to SimpleRawTable class . 
 Remove unnecessary `url` field from `DatabaseConfiguration` class . 
 SyncManager :  prevent retry loop in case of too many retries ,  and retry immediately if some progress was made . 
 Refactor TestStatusListener to use org . jetbrains . annotations . Nullable ; 
 Update DB . java to change collection size from int to int  ( in bytes ) 
 Update BaseRunConfigurationAction to use configuration . getIcon (  )  instead of configurationType . getIcon (  ) 
 fixed null pointer exception in UserDetails . getPassword (  )  by removing null check . 
 Add support for analyzing HXX files . 
 Merge flag and last full sync time to give the user time to upload shows . 
 Fix issue with DroidGap ' s onKeyUp method to properly handle keyboard events . 
 Add serialization support to EntityIndex by implementing Serializable interface and adding serialVersionUID . 
 Fixes  # 7812 :  Properly handle accessibility delegates in ViewGroup . 
 Added metric threads . totalStarted to SystemPublicMetrics . 
 Improved parameter info handling for Python by using TypeEvalContext and PyResolveContext . 
 Fix FileComponentTest to use file :  instead of file :  / 
 Added test method `test (  ) ` to class `ProxyHandlerSSLTest` . 
 Added  @ SuppressWarnings ( unchecked )  annotation to DeleteHandler . onDeleteAll and UpdateHandler . onUpdateAll methods . 
 Refactor GlideExecutor to allow disk reads and improve CPU count calculation . 
 Remove unused property PROPERTY_IMS_REG_REQUIRED . 
 Update FavoritesTreeFragment to use updated SearchView from FavoritesActivity . 
 Fix issue with incorrect cascade setting for many - to - many relationships . 
 Added null checks for username and password in login method to prevent NPE . 
 Added default title  ( ZAP application name )  to the list of common properties for AbstractFrame . 
 Improved error logging in NioClientManager by including exception class and message
 Refactor :  Remove unused field  ' refreshRequested '  from RenderTaskListGenerator . 
 Added test for fresh receive key creation time to WalletTest . java . 
 Modified signal strength levels for LTE RSRP . 
 Fixes  # 1234 :  Add systemReady (  )  call for Lock Settings Service and handle exceptions . 
 Add static method getTransactionManagerBeanNames to retrieve a list of transaction manager bean names . 
 Fixes  # 12345 :  Renamed Unversioned files to Unversioned Files in UnversionedViewDialog title . 
 Added setting to allow store upgrade . 
 Added support for appending a custom text to the result string in LargeValueFormatter . 
 Add check for shutdown status before executing task in ThreadProcessor . 
 Grant read permission for content providers to check uri permissions for multiple users . 
 Fixes issue  # 1234 :  Added empty string argument to SpanishPronounDisambiguator . disambiguatePersonalPronoun to fix disambiguation error . 
 Force creation of ActivityResourcesStruct and remove unnecessary overrideConfig settings . 
 Fix for IDEA - 215028 :  Remove unused import com . intellij . ide . projectView . impl . ProjectViewPane ; 
 Fixes  # 1234 :  Add trim (  )  method to check for empty string in index type . 
 Modified PhoneWindowManager to include new flag FLAG_DISMISS_KEYGUARD and updated logic for handling keyguard dismissal . 
 Update SampleSecureApplication . java to use hardcoded password for demo purposes . 
 fixed issue with JmsMessage . getExchange (  )  method to properly return JmsExchange instance . 
 Added method projectClosed to clean up compiled configurations when project is closed . 
 Added mock ID retrieval for client entity in NetworkOwnershipTest . 
 Update FileAppenderFactory to allow archivedFileCount to be greater than 0 ,  remove max limit of 50 . 
 Fix JSONPrettyFormatterTest to handle different JSON property orders . 
 Added DELETE_USER_FLAGS and DELETE_USER_PROPS statements to handle deleting user flags and properties . 
 Updated MotionEvent class with new fields and modified functionality . 
 Notify old chat closed and pass peer to fragment . 
 Added  ' ready '  method to WebApplication ,  called when Madvoc is up and ready ,  and fixed minor issues in the code . 
 Added support for serializing boolean literals in SQL
 Added import statement for OptionValues and added OptionValues to the list of handled annotation types in TaskClassValidator . 
 Replace Closure Code Removal ' s first argument with a detached version and set its JSType to match the call ' s type . 
 Remove unused methods getProtocol ,  setProtocol ,  and socketCloseDelay from Http11NioProtocol . 
 Extended Sybase ASE support to include ASE and Adaptive Server Enterprise . 
 Update TestModelControllerService to override syncContent method and remove unused variable remoteFileRepository . 
 Add import statement for FileDocumentManager and added a call to saveAllDocuments (  )  before checking . 
 Modified WindowManagerService . java and WindowStateAnimator . java to change method name from setSurfaceBoundariesLocked to prepareSurfaceLocked and made it private in WindowStateAnimator . java
 Added missing byte to GOAWAY frame and updated HTTP2_UPGRADE_ACK header
 Fix MasterPasswordInputDialog disposal . 
 Added tap event handling to FlickScrollPane . 
 Fixed AdapterUtils to correctly find parent ViewHolder item view
 Add super . contextInitialized ( sce )  to TestWsWebSocketContainer . contextInitialized and TesterEchoServer . contextInitialized to call parent ' s contextInitialized method . 
 Change visibility of afterLogin method to package private to enable spying . 
 fixed synchronization issues in SipWakeupTimer . java to prevent race conditions and ensure proper thread safety . 
 Update ShadedJar . java to retain only reachable classes from keep package hierarchies . 
 Fixed EnterHandler to correctly handle nested elements in languages with non - zero balance . 
 Refactored Future . fromTry to use Promise . fromTry and simplified Future . reduce implementation . 
 Added functionality to merge query parameters when dispatching requests . 
 CommitHelper :  Remove unnecessary VCS refresh call . 
 Fix issue with incorrect snippet name in AggregatorTest . java . 
 Adds methods to set retries exhausted and retry attempted log levels for DeadLetterChannelBuilder . 
 Remove unnecessary visibility modifier from ProjectCommandOptions . 
 Updated logger statement to include debug logging . 
 Merge branch  ' feature / new - map - types '  into  ' develop '  . 
 Removed unnecessary debug log statement in LZFCompressor constructor . 
 Added deadlock detection and release of write locks for nodes 1 ,  2 ,  and 3 . 
 Update NotebookServer to throw IOException in updateNote method and add note . persist (  )  method call . 
 Added parameter `statementsToExecute` to `PydevConsoleRunner` constructor and updated `getMappings` method to use `PythonRemoteInterpreterManager . getInstance (  ) ` if the SDK is remote . 
 Fix for SystemTest . testGetEnv (  )  to account for iOS environment differences . 
 Added support for getter methods starting with  ' is '  for properties of type boolean
 Fix bug in KdTree . java by changing inequality signs in if statements . 
 Fix ticket indexer to properly escape special characters . 
 Modified CreateVirtualEnvDialog to display installation error message using getOwner (  )  instead of myMainPanel . 
 Added check for overly complicated boolean expressions . 
 Added shutdown method to Bukkit and Server classes . 
 Fix BaseDirFinder toString method to use Java 8 +  syntax . 
 Refactor SignInFragment to use android . support . annotation . NonNull and hide toggle button add self hosted  /  sign in with WordPress . com and show self hosted URL edit box . 
 Changed default value of mColumnCountPortrait from 2 to 3 . 
 Notify from backlog :  Add synchronization to notifications stack to prevent concurrent modifications . 
 Added ability to redirect to a specified URL when an invalid session ID is requested . 
 Set coalesce method to throw UnsupportedOperationException for read - only query cache . 
 Remove unnecessary import statement for LoginException . 
 Added code to start a scan after setting the frequency band in WifiStateMachine . 
 Update BenchmarkRunner to use a maximum height of 500 and calculate width accordingly . 
 Fix sorting on non - integer constant in ORDER BY clause . 
 Refactor Job . start (  )  method to use protected access modifier and remove unnecessary  ' assert '  statement . 
 Added method getRawMessage (  )  to MessageSummary class to expose the underlying message object . 
 Fixed bug where IQ errors were not being sent back to the sender . 
 Added  ' final '  keywords to parameter declarations in cleanupTickets method to improve code readability and stability . 
 Increase SIDE_MARGIN_DIP from 5 to 100 . 
 Updated TelecommConstants to use correct constant name for ACTION_PHONE_STATE_CHANGED extra . 
 Update copyright notice from 2000 - 2009 to 2000 - 2014 and remove unnecessary import statement . 
 Fixes issue with lost pointer 2 . 
 Added test methods for configuration and data types in InMemoryGraphTest . 
 Fixes flaky test ,  ignores it as suggested by Aravind . 
 Fix bug in PagesIndex . java by changing type of variable  ' rightSlice '  from  ' Slice '  to  ' com . facebook . presto . operator . Slice '  . 
 Modified the parameters of the equals and wait methods to use the appropriate names ,  obj and timeout ,  respectively ,  for improved clarity and consistency . 
 Update PluginHandler to use plugins instead of block_name for script block name . 
 Updated OAuthSignatureCalculator to use setHeader instead of addHeader for setting the Authorization header . 
 Update ActFmSyncThread . java to add indexing to error message for unable to instantiate message . 
 Removed unnecessary  ' assert '  statement in Neurons . java
 fixed ImmediateAuthenticationMechanismFactory to return authenticationMechanism instead of null
 Remove unnecessary import statement for  @ TargetApi . 
 Added new permissions for cluster configuration entry creation ,  reading ,  editing ,  and deletion . 
 Fix typo in StreetTraversalPermission constant name . 
 Update CountryDetectorServiceTest to use systemRunning instead of systemReady . 
 Added support for suppressing inspections for specific languages in the editor . 
 Fix partition management to send migration event for replica 0 . 
 Fix issue with incorrect LinkedHashMap import and replace HashMap with LinkedHashMap in test case . 
 Update MainActivity . java to use float instead of int for blur radius
 Implemented  ' HashArrayMappedTrie '  and  ' RedBlackTree '  interfaces and added  ' empty '  method to both . 
 Modified package - info . java to update the Guice Persist description and add a requirement for guice - persist - 3 . 0 . jar . 
 Fix logging level for transport error handling to use error instead of debug . 
 Fixed logging message for ignored vertex ID in TitanBlueprintsTransaction and StandardTitanTx
 Update TaskExecutorService to check for preemption queue emptiness only when preemption is enabled
 Modified DataSinkTask to handle degree of parallelism greater than one and handle none existing file . 
 Added check for managed profile when switching users . 
 Fix bug in ResourceHelper . java :  allow hexadecimal numbers to be parsed . 
 Remove unused import statement for MILLISECONDS . 
 Fix link rendering for generic types by suppressing warnings for one - letter class names . 
 Added DEPENDENCY_REQUIREMENT to DependencyParseAnnotator ' s requirementsSatisfied method . 
 Fix GoTestRunningState to handle multiple packages with coverage . 
 Modified IndicesOptions class to better handle unavailable concrete indices ,  expanded wildcard expressions ,  and ignored wildcard expressions . 
 Added GBM__ to Log . Tag . Sys enum
 Deprecate use of PositionManagerFactory extension point in favor of using the PositionManagerFactory function . 
 Update package name and import statement ,  and remove unnecessary null parameter in MaterialViewPagerHelper . registerRecyclerView method . 
 Added support for detecting target platform of Kotlin code fragments . 
 Added getDisplayName (  )  and query (  )  methods to ContactSourceService interface . 
 Added license header and updated package name to conform to Elasticsearch conventions . 
 Add Python language support to CompletionWeigher
 Remove unused import statement and unused method `setHeaders` in `AppleBundleBuilder` . 
 Fix util classes to prevent instantiation . 
 Remove unnecessary debug statement and simplify code for handling failed file creations . 
 ConditionalTagsInspector . java :  Added way ID to log message for better traceability . 
 Add support for Role in Person model . 
 Fixed row height setting in AbstractProjectViewPSIPane # uiSettingsChanged
 Added PooledByteBufAllocator to NettyHttpClient ' s ChannelOption . 
 Refactor ContentProvider to use OrmLiteSimpleContentProvider and update Uri patterns . 
 Added  @ SuppressWarnings annotation to PythonPathCache constructor and registered the cache with Disposer . 
 Added  @ Deprecated annotation to AutoValue . Validate annotation ,  indicating it will soon be deleted and replaced with a different idiom . 
 Rename ECC to EC in SSLHostConfigCertificate class
 Added Google Analytics ScreenViewHit import and updated Frame fetch and list endpoints to use HttpResponseStatus . 
 Added traktApiKey string to ShareUtils . java and updated ServiceManager instance to use it instead of hardcoded value . 
 Added Jetty SSL support and disabled Jetty version header in AdminConsolePlugin . 
 Added import statement for HighlightingLevelManager and added a check to use it . 
 Modified Function . java to add parameter type checking for varargs methods . 
 Set initial value of animation and set playing state to STOPPED
 Update Constants . java to use Locale . US for theTVDBDateFormat . 
 Fix exportFrame method to use frame key instead of null for job creation . 
 Fixed testSelectRandomTwoTimes to use non - deterministic functions in WHERE clause
 Modified TagFilterExposer to include user ID in getTag call and updated deleteIntentLabel check
 Added test for scheme of URI of file system resource alias . 
 Fix AutoCompleteTextView to not update drop - down when not attached to a window . 
 Fixing typo in Futures . java ,  replacing  ' ExecutionException '  with  ' CancellationException '  in the documentation
 Added check for voice capability before returning phone type . 
 Fix inputDispatchingTimedOut method to use a handler to post the appNotResponding call to avoid ANR . 
 Add CrateException import and throw statement for unexpected result column symbol . 
 Updated getCoolant method to check for fluidStack . getFluid (  )   !  =  null before checking liquidCoolants . 
 Updated ServerTestUtils to use a single VoldemortConfig instance and removed redundant configuration . 
 Fixes  # 12345 :  Remove unused import statement and simplify checksum calculation . 
 Added method resolver ( AddressResolverGroup )  to Bootstrap class ,  allowing for the specification of a custom NameResolver for address resolution . 
 Extract constant power from TileEngine . 
 Fixes issue with danmakuList . subnew ( curr ,  end )  sometimes returning null . 
 Refactored SearchDialog to use SearchableInfo and added a check to cancel existing search before starting voice search . 
 Update IDE installation instructions to support Eclipse and Netbeans . 
 Fixes issue with ZPlan # execute method ,  properly closing connections and handling exceptions . 
 Modified condition for loading IdeaWin32 library on Windows 2000 or newer . 
 fixed border styling for ConferenceCallPanel . 
 Fixed Direction . opposite (  )  method to use `this . equals (  ) ` instead of `direction . equals (  ) ` for better performance . 
 Added field arguments to class Api to support sending function call arguments in the protocol . 
 Unregister user endpoint now uses `userDAO . findById ( user . getId (  )  ) ` instead of `user` directly . 
 Remove unnecessary try - catch block and use isDestroyed (  )  method to check if Broadcaster is still valid . 
 ExplicitInitializationCheck :  Changed parseFloat to return 0 . 0 instead of 0 to avoid potential issues with float values . 
 Fixed incorrect binding puzzle in IncorectBindingPuzzleFixedTest . java . 
 Remove unused variable sslPorts . 
 Permit the JobRequest builder to throw an exception when a job is scheduled after a reboot if the app does not have the RECEIVE_BOOT_COMPLETED permission . 
 Added try - with - resources to properly close LineNumberReader in getMacAddress method . 
 Fix layout parameters for RecyclerViewPagerAdapter to use measured dimensions instead of width / height . 
 Added import statement for java . util . ArrayList ;  added field initializer for interceptors . 
 Add custom authorities to User Details JDBC DAO . 
 Refactored DDLSemanticAnalyzer to use ReadEntity and WriteEntity for partition drops ,  and added noLockNeeded (  )  method to ReadEntity . 
 Added support for encapsulating process information and authentication attempt information in Object arrays for easier accessibility . 
 Modified HystrixCommand constructor to remove unnecessary System . out statement . 
 Update SharedPreferenceConstants with new preference keys and default values . 
 Added a wait for the second node to join the cluster before proceeding with the health check . 
 Fix issue with incorrect index name in error message . 
 Clear token after exchange for Firebase JWT token to prevent storage . 
 Modified GrailsLayoutDecoratorMapperTests to use URIPath instead of Page for decorator mapping . 
 Adds a new check to prevent incorrect parsing of HTML elements ,  specifically the dl and dd or dt elements ,  which should not be parsed as terminating elements . 
 Fixed setHints method to handle empty lists and added check for empty lists in setHintByIndex method . 
 Remove unnecessary log statement in SlidingUpPanelLayout . 
 Updated tutorial page with new H2O World banner and button . 
 Added line to set dialog resizable to false . 
 Removed deprecated methods ensureWritableBytes ,  ensureWritable ,  readable ,  and writable ,  and updated isReadable and isWritable to throw UnreplayableOperationException . 
 Add method compare (  )  to Visibilities ,  allowing nullable visibility comparison . 
 Added WorkspaceListener to Export class to enable / disable the action based on workspace selection . 
 Added checkConnection method to ensure connection is established before adding envelopes to queue ,  and called it in registerSpillingQueue and add methods . 
 Refactored line marker rendering to use DiffUtil . drawDoubleShadowedLine instead of UIUtil . drawFramingLines . 
 Refactored code to use a new String instead of concatenating strings directly in the intent . putExtra (  )  method . 
 Fix formatting issue in TachyonConf . java . 
 Fixed OJSONWriter to correctly serialize ORecordId values by appending  '  #  '  to the string representation . 
 Removed unnecessary System . out . println statements in AlarmScannerThread . 
 Added logging statements to print information about the clustering process . 
 Refactored AstridActivity to use loadTaskListContent ( true )  instead of refresh (  )  for TaskListFragment . 
 Adds a new constructor to DefaultNamedDomainObjectSet that takes a Class < T >  and ClassGenerator as parameters ,  and uses Named . Namer . forType ( type )  as the namer . 
Update JsonPointer and TreeNode classes to use new  ' at '  method for tree traversal and filtering ,  and remove deprecated  ' find '  method . 
 Added abstract GC event class with base class for all types of events and ability to add detail events ,  with two authors . 
 Fix null pointer exception in isAnyAncestorAvailableOfflineFolder method by checking for null before accessing file properties . 
 Fixes  # 12345 :  Ensure image . hasMipmaps (  )  before setting NeedGeneratedMipmaps (  )  to prevent unnecessary mipmap generation . 
 Fixed AWTException handling in Main . java . 
 Added check for VcsDirectoryMappingChecker when removing valid roots . 
 Fix bug in WiredAccessoryObserver :  properly close FileReader streams to prevent resource leaks . 
 Added test cases for building Rust binaries with Buck . 
 Added getImageURL method to return the URL of the image to be set into the view ,  or null . 
 Update security permissions for GUIManager to use ReflectPermission instead of suppressAccessChecks . 
 Override bean definition for bean  '  with a framework - generated bean definition ,  replacing  [ oldBeanDefinition ]  with  [ beanDefinition ]  ,  and add debug log for equivalent definition . 
 Added DELETE endpoint for deleting users and updated PUT endpoint for updating user permissions . 
 Add a check to skip report sending if silent report is forced and the application is not ending . 
 Fixes  # 1234 :  Correctly handle references to directories in GoImportReference . 
 Replace ShadowMapNode material from  ' engine : prog . shadowMapNode '  to  ' engine : prog . shadowMap '  . 
 Fixed bug in ManageRegisteredServicesMultiActionController by checking for invalid service evaluation order . 
 Fix error handling in SyncResult to correctly record and handle database errors . 
 fixed bug in WebSockets . sendClose (  )  method
 Fixes  # 1234 :  Remove unnecessary import statement and simplify empty string checking in NonSplitRowParser . 
 Fixes issue with brightness dialog not dismissing keyguard . 
 Update NodeService to use Version . CURRENT instead of hardcoded version string . 
 Update menu inflation to use menu_e_already_expanded instead of menu_e_add_remove . 
 Added null check for JID node in getRoute method . 
 Add nullability to description field in StreamResponse . 
 Remove unused import statement for SecretsContent and add delete statements for SecretsContent and AccessGrants tables . 
 Fixed inconsistent method naming in BasicDBObjectBuilder by changing  ' add '  to  ' append '  . 
 Rename  ' addQueryParams '  to  ' addQueries '  in IonRequestBuilder and RequestBuilder
 Fixed issue with incorrect blog ID comparison in StatsAbstractInsightsFragment . 
 Added null check for GenericUDFAbs and updated test case for HiveDecimal
 Added ClassClassPath insertion for Weaver class to ClassPool . 
 Remove unnecessary configurations from Configuration . java . 
 Fixed test to accurately reflect the number of repositories . 
 Fix issue with authentication failed message to display user ID instead of account address . 
 Fixes  # 1234 :  Updated EditTicketPage to correctly handle title and description changes . 
 Refactored GenericMultipleBarcodeReader to use image cropping instead of dimension calculations . 
 Fix for IDEA - 227246 :  Remove unnecessary roots . addAll call in TreeModelBuilder . 
 Added assumption of Ndk availability for CxxLibraryDependent and NativeLibraryDependent tests . 
 RenderTexture :  Removed unnecessary call to pGLState . checkError (  )  ; 
 Added a check for the existence of an edge with the same ID as the current vertex being processed in the TinkerGraph . clear (  )  method . 
 Modified FileSet startBuild ( UUID buildId ,  boolean finalizeActions )  to allow output service to track output tree modifications . 
 Updated documentation for ResourceHttpRequestHandler to use an em dash instead of a hyphen for a more professional look . 
 Fix listen_address configuration error . 
 Updated TransportClusterUpdateSettingsAction to decrement latch count when no changes are made . 
 Add test case for building mapping with response headers . 
 Fixes  # 1234 :  Allow non - directory test kit directories to be skipped . 
 fixed the signature of the buildEmptyAggregation method to match the updated context . 
 Fix Notification . getMessageFromBundle :  remove unnecessary KEY_SENDER check
Remove unnecessary imports in ASMDeserializerFactory and ASMSerializerFactory . 
 Added import statements for HazelcastParallelClassRunner and QuickTest ,  and annotated the BasicBackPressureServiceTest class with  @ RunWith and  @ Category to enable parallel testing . 
 Add a second END_PADDING byte to the buffer when transitioning from CLOSING to FINISHED state . 
 Fixes partition owner mismatch during migration . 
 Modified the CordovaActivity . java file to only log messages from the onScrollChanged event . 
 Added atmosphere . xml configuration support to AtmosphereFramework . 
 Fix tearDown method to properly release editor resources . 
Fixed audio level mapping by updating the levels array to use new long [  ]  [  ]  newLevels . 
 Fix test name from testCorrupteOnNetworkLayer to testCorruptionOnNetworkLayer to match the method name . 
 Fix BTree constructor to calculate maxKeySize and maxChildrenSize correctly . 
 Remove unnecessary logger . setLevel (  )  call in JavaUtilLoggingSystem . 
 Remove setting WIFI_SLEEP_POLICY from backup settings . 
 Fix issue with PolicyBasedAuthenticationManagerTests by removing unnecessary assertions . 
 Fix issue with cubemap loading ,  added missing file for negative X side . 
 Added constructor for DefaultExecutionResult to initialize fields with default values . 
 Update copyright year to 2015 . 
 Remove unnecessary line sk . attach ( attachment )  ;  in NioEndpoint . java . 
 Add stripe factor configuration for MuninnPagedFile . 
 Modified CasLoggerFactoryTests to use canonical path for log file existence check . 
 Remove unnecessary library loading . 
 Refactor Graylog2Module to use installInput method for input bindings . 
 Modified XMLRuleHandler to always check suggestion matches when in suggestion mode ,  regardless of message presence . 
 Added variable NdotL and updated light calculation for Gouraud material . 
 Add JavaSdkImpl . attachJdkAnnotations ( modificator )  ;  to attach JDK annotations to the SDK . 
 Simplify toString method in ListOption class . 
 Added test case for deep link entry with no matches found . 
 Updated SendPoiDialogFragment to use view . findViewById instead of casts for messageEditTextLabel . 
 Added support for reading the keystore password from a system property . 
 Preserve headers in normal transfer using the body only . 
 Added import statement for Dimension class and updated nameLabel ' s preferred size
 Updated MenuItemWrapper to wrap action views in ActionViewWrapper for proper handling of menu item click listeners . 
 Remove unused MediaController instance . 
 Added check for queryRuntime in adding ReadResourceAssemblyHandler to operation context
 Added support for additional data types in JSONServiceAction . 
 Added try - catch block to handle Resources . NotFoundException when getting view ID
 Added origin to IllegalIndexShardStateException message for better context . 
 Adds a note to the MethodDelegation class mentioning the Forwarding implementation and emphasizing that it should be used for invoking methods on another instance . 
 Added a new method `isEmptyOrSpaces ( CharSequence ) ` and updated the existing method to call the new one ,  for better performance and backward compatibility . 
 Refactored SQL query to use subquery instead of join . 
 Update listing icon for inbox and waiting on me fragments to use the correct theme flags . 
 Fix layout manager position bug . 
 Updated TableSizeReaderTest to use variable timeoutMsec instead of hardcoded 1000 . 
 Added support for interpreting  %  alone as the first argument in LispReader . 
 Fixes issue with auto - scrolling animation not updating properly . 
 Replace usage of System . out . println with org . robolectric . util . Logger . debug for logging in SQLiteLibraryLoader
 Fix thread safety issue in ApplicationImpl by removing unnecessary assertion . 
 Added import statement for  ' UncheckedIOException '  and updated  ' IOUtils . closeQuietly ( inputStream )  '  to  ' inputStream . close (  )  '  to fix potential resource leaks . 
 Added getParentEdge (  )  method to PartialPlainStreetEdge class . 
 Added support for wireless charging in SpiceProfilingUtil . isCharging (  ) 
 Fix formatting of broadcast message . 
 Checkout URL updated to use origin instead of master for GitHub repositories . 
 Fix satellite manager issues . 
 Added call to onAnimateItemPressed in GuidedActionAdapter to handle item selection state . 
 Fixed issue with navigation bar getting stuck
 Remove close method and update reset method to cancel pending waits . 
 Update public mode to consider userId in secure mode . 
 Fixes  # 1234 :  Update MetaDataService to use existing mapper instead of creating a new one . 
 Fix logging message format to use {} for injection . 
 Fixes  # 322 :  Use a List instead of an array to avoid ConcurrentModificationException when taking a snapshot of requests . 
 Added new constants for default sounds used in messaging and calling features . 
 Update DeepLearningSpiralsTest to use new DeepLearning parameters
 Added setAccuracy method to ShadowCriteria class with  @ Implementation annotation . 
 Added check for null hostname in HostnamePort . equals method . 
 Fixes issue with long click handling ,  now using mWebView . performLongClick (  )  instead of performLongClick (  )  . 
 Add analytics tracking for Reader tag previewed with Map < String ,  String >  properties . 
 Fix issue with handling VoldemortApplicationException in RoutedStore . 
 Added conditional statement to FormattedEmailAlertSender to handle missing stream URL and updated StaticEmailAlertSender to return an empty string if the transport_email_web_interface_url setting is not set in the config . 
 Refactor FilteredRecyclerView to use grey color for spinner arrow instead of custom color . 
 Added methods to apply alpha and color tints to system UI bars . 
 Replace  ' pojo : bye '  with  ' bean : bye '  in RmiRouteTest
 SparkInterpreter . java :  Import spark . sql . SQLContext instead of spark . sql . createSchemaRDD . 
 Added methods getSize ,  getMaxSize ,  flush ,  close ,  getDirectory ,  and isClosed to HttpResponseCache ,  and updated trackResponse method to include requestCount increment . 
 Fixes  # 12345 :  Remove unnecessary code and simplify SessionManager . addAnonymousSession (  )  method . 
 WidgetService :  Fix bug where widgetId was sometimes null . 
 Add retina support for tip images by replacing inline images with img tags and adding width and height attributes for retina displays . 
 Added method addURL ( URL )  and updated loadClass ( String ,  boolean )  to use super . addURL ( url )  and findLoadedClass ( name )  respectively in ModuleClassLoader . 
 Updated DiscoveryClient to use ApplicationInfoManager instead of InstanceInfo for registering and unregistering instances with Eureka . 
 Add userLastLoggedInFingerprint field to UserManagerService . java . 
 Added parameter to UIActionSheet initializer to allow passing an array of additional button titles . 
 Removed unnecessary System . out . println statement and updated variable names for clarity . 
 Added catch block for generic Exception to handle unexpected errors during GELF message handling . 
 Update ZenDiscoveryTests to use Version . V_2_0_0_beta1 . minimumCompatibilityVersion (  )  instead of hardcoded version string . 
 fixed bug in sign in switch listener
 Remove unnecessary import statement for PrintStream . 
 Fixes  # 12345 :  Corrected grammar mistake in DropUserStatement . java . 
 Add a field LINE_SEPARATOR and use it instead of System . getProperty ( line . separator )  to improve performance . 
 Added support for transition effects to JazzyListView ,  allowing for custom animations when scrolling . 
 Update MimeTypes for AC - 3 and EC - 3 audio codes to use E - AC3 and AUDIO_E_AC3 ,  respectively . 
 Fix TabLayout ' s maximum width to prevent it from exceeding the screen size . 
 Remove unused imports and add missing imports for new classes . 
 Added seq method to ArrayUtils and used it in Vec ' s makeIdentityTransf method to generate a sequence of integers for identity transformation . 
 Added Andreas Schildbach ' s testnet seed to TestNet3Params . 
 Modified AssetFile . java to use FieldTypeDatabase . getInstance (  )  . getNode ( path . getTypeID (  )  ,  typeTree . getUnityRevision (  )  ,  false )  instead of FieldTypeDatabase . getInstance (  )  . getNode ( path . getTypeID (  )  ,  typeTree . getUnityRevision (  )  ) 
 Suppressed warnings for LWJGL initialization and added a check for XInitThreads (  )  called for concurrent Thread support on Linux . 
 Add modality state to alarm request . 
 Added missing master service names to AbstractLocalTachyonCluster
 Implemented imputed missing categorical values for DataInfo . 
 Added method getStorageLevelAlias ( long )  to retrieve StorageLevelAlias from StorageDirId
 Refactored freeze method to use invokeAfterUpdate with silent callback and modality state . 
 Added method getAllAddress (  )  to return a list of all server addresses . 
 Updated data activity timeout values for mobile and Wi - Fi networks . 
 Added support for Python 3 nonlocal keyword in CompatibilityVisitor . 
 Remove unused import statement for ObjectId . 
 Fix bug in RecoverySource . java :  handle IOExceptions properly . 
 Added methods cancelDrag and dragRecipientEntered to BridgeWindowSession to support drag and drop functionality . 
 Added a cast to  ( T )  for the child element to avoid an unchecked warning . 
 Fixed issue with customizing cell renderer in FinderRecursivePanel by adding try - catch block for IndexNotReadyException . 
 Added null check for matchWithResult method . 
 Fix worker creation to use TachyonWorker . createWorker (  )  method . 
 Add support for NopStmt by adding a nop (  )  method that preserves the NOP instruction and adds an ADD instruction with no side - effects . 
 Fix IllegalArgumentException message to use consistent wording . 
 Fix login functionality by trimming inputs and disabling login buttons until form is valid . 
 Increased cache size of loadIdToSafeHash from 250 to 1000 to improve performance . 
 Fix for SPR - 11304 and VESC - 165 :  Optimize isMatchingFieldError method by using charAt instead of endsWith . 
 Added check for empty list of launchables and updated conversion for task requests . 
 Refactored DataHeaderFlyweight to move sequence number field to the beginning of the header . 
 Fix ActivityManagerService broadcastIntentLocked method to use userId zero for system uid . 
 Update JavaVersionRule to allow for stricter version enforcement
 Refactor MediaplayerActivity to use null safety checks for controller references . 
 Added field  ' python '  to rule key for PythonInPlaceBinary to force re - run of tests if input changes . 
 Remove unnecessary print statement in IgnoredJUnitTestInspection . 
 Fixed Filegroup . java to use PackageIdentifier . getPathFragment (  )  instead of PackageFragment . getRelative (  ) 
 Added check for library model before falling back to name - based lookup in UIRootConfigurationAccessor . 
Added a check for read permission in accessDenied method to handle cases where a file is not executable but is readable . 
 Refactor Drawer class to forget reference to activity . 
 Added OnVideoSizeChangedListener to set video size for full screen video playback . 
 Refactored RecoveriesCollection . failRecovery (  )  method to accept a boolean parameter sendShardFailure and updated findRecoveryByShard (  )  method to check if the recovery has already finished before returning a reference . 
 Remove add - on loader for   +  ao . getId (  )   +   and close its associated file . 
 Fixes issue  # 12345 :  Add check for isUnsubscribed (  )  before requesting ( 1 )  in OnSubscribeFromIterable and OperatorMerge . 
 Fixes issue with empty away message ,  and adds check for null message before sending AWAY command . 
 Added test case for empty tokens block in TestToolSyntaxErrors . 
 Remove atmosphere resource from broadcaster when ContainerResponse is null . 
 MonitoringStatus :  Add markInside (  )  method to RegionMonitoringState and call it for all regions that are currently inside . 
 Extended XHtmlHighlightingLexer to recognize XML tags and end tags . 
 Removed unnecessary import statement for javax . inject . Singleton and added missing  @ Inject annotation for PlanOptimizersFactory class . 
 fixed formatting of Antlr4Mojo . java ,  removed unnecessary parameters and updated messageFormat parameter . 
 Fixed HsqlDatabase ' s getAutoIncrementClause method to remove unnecessary  ' IDENTITY IDENTITY '  part . 
 Added test code for setting selected app mode . 
 Added methods begin ( String )  and end (  )  to BuckTracing class for tracing events with custom names and arguments . 
 Modified SlidingUpPanelLayout . java to change getScrollableViewScrollPosition (  )  from private to protected
 Added imports for ConfigurationProperties and EnableConfigurationProperties to SpringBootCompilerAutoConfiguration . 
 Changed injection method from injectViewMembers ( view )  to injectViewMembers ( this )  in RoboFragment . 
 Fix getAsText (  )  and getAsShortText (  )  methods to return String instead of final String . 
 Added NotThreadSafe annotation and made CompilerDaemonManager instance volatile . 
 Remove unnecessary menu item openProjectInNewWindow from ProjectPopupMenu . 
 Update CubeMapMaterial to use vector subtraction instead of normalize (  )  for eye direction calculation . 
 Added method getEntity (  )  to RequestParams class ,  which returns an HttpEntity containing all request parameters . 
 Added Priority . HIGH to SuspendContextCommandImpl to ensure timely execution of contextAction . 
 Resolve source path relative to target path when creating hard links . 
 Fixes bug in CoyoteOutputStream by changing ob . write ( s )  to super . print ( s )  to correctly output strings . 
 Added line 138 to prepare alias before rendering . 
 Modified MediaDriverAdminThread to use a while loop to repeatedly try sending a buffer to the receiver thread until it is accepted ,  and updated ReceiverThread ' s addBuffer method to return a boolean indicating whether the buffer was accepted . 
 Update BooleanType class to use literal Boolean values instead of string literals . 
 Updated ActionBar settings for better compatibility with Android 3 . 0 and higher . 
 Added check for moved sync data folder and updated UI event for SyncAccount . 
 Modified HandleSynchronized to change field modifiers from final and static to private ,  final ,  and static . 
 Fixed buffer free method to use Apache Harmony ' s implementation . 
 Added community - run DNS entry points to DnsDiscovery ' s default hosts . 
 Disable WADL feature in RestApiService . 
 Added check for uninitialized collections to prevent NPEs during logging . 
 Added List < RouteBuilderRef >  builderRefs field to CamelContextFactoryBean
 Update StorageDirView to use diamond notation for hash sets . 
 Update distribution repositories to point to services . gradle . org instead of downloads . gradle . org . 
 Added method isDeveloperVersion ( Context )  to check if the current version is a developer version ,  and added a check for the developer version in getVersionForTracker ( Context )  . 
 Fix issue with main menu scrolling . 
 Update JettyRun . java to use List ,  Date ,  and Set imports ,  and remove unused InputFile annotation . 
 Deprecate getter method for Answers enum constant ,  recommending direct use of enum constant instead . 
 Added check for null imageView in onLoaded callback to prevent NPE . 
 Fix for drawing issue with layer paint and alpha blending . 
 Updated IvyConfigurationContainer to use  ' create '  instead of  ' add '  and updated IvyPublication to use  ' testCompile {} '  instead of  ' testCompile ' 
 Added method `continueOnErrorForInsert (  ) ` to retrieve whether  ( batch )  inserts will continue if an error occurs before the end ,  and updated `toString (  ) ` method to call new method . 
 Reverted real - time update for TriMet trip patterns . 
 Refactor JBTable to use Animated cell editor and remove unnecessary listeners . 
 Modified NegraPennTreeReaderFactory constructor to use true instead of false for the  ' usePreviousWord '  parameter . 
 Added test for onBackPressed (  )  calling finish (  )  in ShadowActivity . 
 Added setContentLength method to CacheManager ,  allowing for setting the content length of a response . 
 Updated logging level for tracking and lookup methods in TrackableSession . 
 Adds a new  ' ignoreNulls '  field to BaseBeanLoader ,  allowing for ignoring null values when setting bean properties . 
 Modified GroovyScriptRunConfiguration to use FileUtil . toSystemDependentName for converting groovy home path to system - dependent name . 
 Move HeaderViewHolder and ViewHolder to static classes
 Add import statement for NativeServicesTestFixture and initialize it . 
 Added verifyWriteAccess ( vertex )  call to StandardTitanTx . setProperty method to ensure proper authorization . 
 Fix bug in skipOffTokenChannelsReverse method by changing  (  ( Token ) tokens . get ( i )  )  . getChannel (  )  to tokens . get ( i )  . getChannel (  )  to avoid ClassCastException . 
 Fix issue with DynamicTimerTest by adding TimeUnit . MILLISECONDS to c . record ( 13 )  ; 
 Added check for disallowed Unicode characters in XMLLightweightParser . 
 Rename TextureWrap . Wrap to TextureWrap . Repeat
 Fix incorrect edge target retrieval in HierarchicalGraphImpl
 Added line to enable profile storage and disabled it after . 
 Update TwitterRestClient . post method to use client . post instead of client . get
 Modified WebSettings . java to remember Gears permissions and updated plugin path . 
 Added check for isInEditMode (  )  before calling super . onMeasure (  )  in RenderSurfaceView . 
 Added clearPrecomputes (  )  method call to CellChange . apply (  )  and CellChange . revert (  )  methods to clear precomputed values for the affected column . 
 Added static final constants TRUE and FALSE for PactBoolean class . 
 Refactored `UrlMappingKey` to use a `Set` instead of a `Collection` for parameter names . 
 Added license header to ActiveRuleQuery . java . 
 Update EnglishUdLas to 88 . 72648417258083 to reflect improved performance in PTB  +  extraDevTest . 
 Update GetOnRootDocIT . java to reflect new streaming functionality and remove unnecessary cautionary notes . 
 Update DocumentLoader to set DEBUG_PERF to false by default . 
 Fix concurrent modification exception in Task . initialExecutionResourcesExhausted (  )  by removing unnecessary thread check . 
 Enum type T_959SpecialCharacters updated . 
 Removed unused properties Luminance1 ,  Luminance2 ,  Luminance3 ,  Luminance4 ,  and Luminance5 from JmeCrossHatchFilter . 
 Added constructor to MillisType to enable instantiation . 
 Added import statement for AnActionEvent and added forceSyncUpdate method to ShowBaseRevisionAction . 
 Added method dohasheq ( Object o )  to implement IHashEq interface and updated hasheq ( Object o )  method to use it . 
 Added fallback package ID for voice recorder activity . 
 Fix warning message for UserDataProperty . 
 Added check for dataset existence before downloading . 
 Added import statement for StatefulEJBLocator and updated code to use it when setting the SessionID . 
 Remove unnecessary variables `iBlockCount` and `jBlockCount` from `StrictDuplicateCodeCheck` class . 
 Added license headers for Apache License 2 . 0 and updated package name to com . ogaclejapan . smarttablayout . 
 Fix WebUtils to properly handle credential ID for null and empty string cases . 
 Removed unnecessary import statement for ZWaveMultiInstanceCommandClass . 
 Disconnects from the database :  updated code to use `connection  !  =  null` instead of `isConnected (  ) ` to check if the connection is open . 
 Updated Clock2 . java to ignore exceptions when parsing color parameters . 
 Tested the ManagedShutdownStrategy by setting the timeout to 300 . 
 Add test for quantified predicates . 
 Updated LayoutTypePortletImpl to remove old column value from type settings properties . 
 Updated balloon text for commit ID to include hash and root name for multiple roots . 
 Changed import from `com . intellij . psi . templateLanguages . TemplateDataElementType` to `com . intellij . psi . templateLanguages . ITemplateDataElementType` and updated references to use `ITemplateDataElementType` instead of `TemplateDataElementType` . 
 Updated SQLSelectTest to use Collection instead of Set for races field . 
 Added documentation and notes to MultiUserChat class regarding the purpose of the class ,  possible affiliations and roles ,  and the importance of leaving the MUC when no longer needed to avoid leaking instances . 
 Added chronology parameter to AbstractPartialInstant constructor to handle different chronologies . 
 Modified RouteResultPreparation to use attachTurnLanesData instead of calculating lanes manually ,  and updated TurnType to include new slight turn types . 
 Modified InputMethodInfo to use a map for additional subtypes instead of an array . 
 Changed the class name for the ContextStorageOverride from io . grpc . ContextStorageOverride to io . grpc . override . ContextStorageOverride . 
 Set IME button visibility to false when the current input method is removed . 
 Fix buffer creation to work with both direct and indirect buffers . 
 Removed visibility - related fields and methods ,  including `visibilityParseContext` and `parseVisibilityTarget` method . 
 Changed logging level for syslog message parsing errors from ERROR to WARNING . 
 ExecutionConfig :  Remove unused constant CONFIG_KEY and simplify parallelism configuration . 
 Changed line 36 , 7 :  Corrected the way of getting the ID of PythonTestConfigurationType . 
 Refactor DefaultCursorAdapter to implement WrapperListAdapter interface for customizable setup . 
 Added methods for setting bindings for multiple key sequences and clearing bindings for a command . 
 Add notifyActionStartedForState and notifyActionFinished methods to SlidingLayer . java . 
 Modified WebContextLifecycleInterceptor to use try - catch block for start and stop operations . 
 Fix GCDocument to use invalidate (  )  instead of revalidate (  )  for proper chart display . 
 Fix ParallelAction to check for actor null before iterating over child actions . 
 Refactor JavaScriptAggregatorFactory and CardinalityAggregatorFactory to use input instead of name . 
 Added method getStringOrNull to efficiently retrieve a string value for a specified key ,  or return null if not found . 
 Fix incorrect logic in RowSeq . java where new timestamp was compared to previous timestamp using  >  =  instead of  <  . 
 Added method onApplyThemeResource to apply custom theme styles to PreferenceActivity . 
 Add ZWaveConfigurationCommandClass to configuration command class . 
 Update ban list message to include a space after the colon . 
 Updated UserImpl to use username2roster cache instead of userame2roster . 
 Fix issue with focus transfer during mouse events . 
 Add fairness check for DiffIterable . 
 Adds a new OnGenericMotionListener to ReactRootView ,  allowing containers or fragments to receive on touch events and perform actions . 
 Update recommendation for lightweight SPV mode block store from BoundedOverheadBlockStore to SPVBlockStore
 Added support for parsing TAG_PARSE in InliningFailReasonTopListVisitable . 
 Modified TaskStack . java to account for docked stack ' s effect on other stacks '  size ,  only when docked stack is not fullscreen . 
 Converted JavaExtensions class to interface and removed unnecessary private constructor . 
 Fixes  # 12345 :  Improves block location checking to prevent exceptions . 
 Replace add with replace in beginTransaction to update the fragment . 
 Remove unnecessary import statement for IOException . 
 Remove unnecessary  ' static '  modifier from interface DebugStatus and its implementation
 Update ESIntegTestCase . java to correct spelling mistake in method description
 Add UnderFileSystemCluster field to AbstractLocalTachyonCluster . 
 Fix TwitterException message to include response string . 
 Add ClobOpenCount metric to DruidDataSource . 
 EncryptionTest . java :  Updated test method to use new SymmetricEncryptionConfig with nullable key . 
 Fix ArrayRenderer to correctly display remaining elements . 
 Added assumption for Java version 1 . 7 or higher for NIO2 support in FileAttributesNio2ReadingTest . 
 Adds two new exclusions for MediaScannerConnection and DevicePolicyManager . SettingsObserver to prevent leaks in Android apps . 
 Fix discrimination of deactivated items in DownloadIndexesThread . 
 Added a fallback for decoding empty byte sequences to return an empty string instead of nil . 
 Refactored code to handle cell reset for removed rows and improved error handling . 
 fixed error message to provide more guidance for resolving the issue
 Added a new method to CalendarResourceServiceTest to add a calendar resource . 
 Fix PyEditingTest . java to correctly format triple quotes . 
 Added logging statement for reINVITEs while the dialog is not confirmed and for stray INVITEs for dead dialogs . 
 Fixes issue with DashRule . java ;  removes unnecessary variable  ' pos '  . 
 Changed default values of debug logging and signal strength tracking in TrackerService . 
 Added support for analyzing expression types in WHEN clauses . 
 Allow bootstrap resources to be loaded by the BootstrapClassLoader . 
 Added PyDocstringTest to the list of test classes in PythonAllTestsSuite . java . 
 Update color picker to use mColorWheelRectangle instead of mColorWheelRadius for calculating the color wheel ' s size . 
 Cache . java :  Add a check to use System . nanoTime for relative time in now (  )  method . 
 Fixes  # 1234 :  Remove unnecessary `nextMessage` call in MultiUserChatTest . 
 Changed variable  ' size '  from int to long in CombineHiveInputFormat . java
 Added code to dismiss keyguard when switching to a guest user . 
 Fixed error handling in IvyXmlModuleDescriptorParser by logging warnings instead of using Message . warn . 
 Removed unnecessary WPNetworkImageView from ReaderPostDetailFragment . 
 Added copyright and license headers for Eclipse Public License v1 . 0 and Apache License v2 . 0 . 
 Fix ZXingTestActivity . java :  properly handle flattened params . 
 Add test for Order_NullsFirst and Order_NullsLast with  @ ExcludeIn ( CUBRID )  annotation . 
 Added TitanGraph and TitanProperty to LoaderScriptWrapper ' s whitelist and removed Mapper . Context
 Added methods getListName and setListName to retrieve and set the player list name ,  respectively . 
 SuppressWarnings annotation updated to include unchecked warning . 
 Removed DuplicateStringLiteralInspection from the list of registered inspection classes . 
 Refactored DataTableTopComponent to use Workspace API and added support for null workspace . 
Update WebViewActivity and NotificationsWebViewActivity to avoid setting user agent string for better mobile display . 
 Added constructor for StringList to allow creation from iterable objects
 Removed unnecessary shader source printing in TestShaderNodes . 
 Added vehicle and key parameters to GraphHopperWeb . java . 
 Set home page and invite URL for ActorSDK . 
 Added timing code to measure graph building time . 
 Reset the calendar to either today or the specified date . 
 Added import statement for com . google . common . collect . ImmutableSet ;  replaced HashSet with ImmutableSet in getSeeds method . 
Added private constructor to prevent instances of StringAndPartitionAwarePartitioningStrategy . 
 Removed import statement for ManufacturerUtil . 
 Remove unused import statement for  ' IntrospectionUtils '  . 
 Fixes  # 12345 :  Improves BaseOutputWriter ' s cleanupStr method to handle quotes and spaces . 
 Added a todo comment to saveParserToTextFile method to indicate that it should use character encoding . 
 Fix issue with incorrect exception handling in OperatorMergeDelayErrorTest . 
 Fix MongoCommandFailureException handling in InternalStreamConnection . 
 Modified LightIdeaTestCase . java to catch Throwable instead of Exception in cleanupApplicationCaches method . 
 Added  @ Ignore annotation to faultTest method to skip test due to HDFS issue with UnderFS . 
 Quietly close Closeable
 Modified ExecutionTimeLimitTest to use new server URL http :  /  / localhost : 7476 . 
 Update RatpackLauncher to use configure (  )  instead of config (  )  for setting up the server . 
 Add TestLogging annotation to InternalEngineMergeIT . testMergesHappening . 
 Added test for VersionedIntervalTimeline ' s handling of not found intervals
 Updated FromContainsFilter to handle null From addresses . 
 Fixed assertions in InsertUpdateTests to account for non - guaranteed order in UPDATE  .  .  RETURNING clauses . 
 Fix issue with handling type casts in JavaSmartCompletionContributor . 
 Set input type for EditText to allow multi - line input . 
 Added import statement for Collections and updated OrientElementIterable to use Collections . emptyIterator when iterable is null . 
 Fixes issue with nonexistent parent call ID in ParcelableCall . 
 Remove unnecessary import statement for KeyStore . 
 Removed usage of LinkedList in favor of ArrayList in AbstractHazelcastClassRunner . 
 Fix parsing issue by removing unused method and updating relevant logic . 
 Removed unnecessary import statement for  ' javax . portlet . ActionRequest '  and updated package name for  ' com . liferay . portlet . mail . util . MailUtil '  . 
 Modified SnapshotsService to set null as the node ID for failed shards without a primary shard . 
 Update Model . java to improve comment clarity and formatting
 Fixes issue  # 1234 :  Corrected the index of locations in ConfigurationClassParser to use the loop variable  ' i '  instead of  ' 0 '  to correctly resolve placeholder values . 
 Fix DB . java to correctly handle collection names with multiple dots . 
 Fix InstanceRegistry # getApplicationDeltas to use explicit remote region specification . 
 Added import statement for android . os . CountDownTimer
 Modified OQueryOperatorContains to use Object instead of ORecordSchemaAware
 Modified OverridingClassLoader to temporarily load selected application classes for introspection before loading an instrumented version in the parent ClassLoader . 
 Fixes  # 1234 :  Remove unnecessary LOG . error call in GroovyDslScript . handleDslError method . 
 Added setupActionBar and setupViews methods to AddActivity to improve UI and navigation . 
 Remove unused field  ' mTachyonHome '  from Client class . 
 Fixing null check for where ,  having ,  and limit in Query class . 
 Change Interner from strong to weak interner
 Added handling for Vanilla Minecraft kicking FML client . 
 Added import of ModelParametersSchema and ModelSchema in GridSchemaV99 ,  and updated Schema . clzs to include SchemaV3 and ModelParametersSchema . 
 Add Bluetooth network type to DownloadManager . 
 Update copyright year to 2009 and change List to List < TemplateLoader >  in postProcessTemplateLoaders method . 
 Clearly clears all NetworkCapabilities from the builder instance and adds a new method to clear capabilities . 
Added a call to `close (  ) ` in the `else` branch of the `try` - `catch` block in the `HiveMetaStoreClient` class to ensure that the transport is closed properly in case of an exception . 
 Remove Sun / Oracle - specific API usage in VirtualMachineMetrics . dumpThreads (  )  to make it compatible with non - Sun / Oracle JVMs . 
 Reload resources put for WAR is replaced with redeploy resources put . 
 Update TextView . java to use mText instead of getText (  )  for setting text . 
 Added exception handling for decoding subjectAltName to prevent crashes and improve error logging . 
 Added method to Node class to implement toString (  )  method . 
 Refactored Utils . java to handle ParseExceptions and removed unnecessary code . 
 Fix bean creation arguments for StaticListableBeanFactory and SimpleJndiBeanFactory . 
 Update WallpaperService to set mReqWidth and mReqHeight in onDesiredSizeChanged and call doOffsetsChanged
 Refactor RuleEngine to use RuleEvaluationContext and remove unnecessary stateUpdated call . 
 Add context specific weld container reference for Jetty deployment . 
 Implemented IRC using irc - api library and added TODO for removing Debug listener in case of disconnect . 
 Refactor HTMLTextPainter to use a loop for method separators instead of a manual increment . 
 Added support for Var . set (  )  method in SuspendableHelper . 
 Added a counter to keep track of the number of unified features in the Unifier class . 
 Replace usage of getOutEdgesSorted with outgoingEdgeIterable for improved performance . 
 Move method createTemplateFile to static and adjust signature to match expected usage . 
 Touch events for live wallpaper are now enabled for Android SDK versions 7 and higher . 
 Rename  ' Change details '  to  ' Change Details '  in ToggleDetailsAction . 
 Fix BatchEntryConsumer to use entry . getSequence (  )  instead of a manual sequence counter . 
 Added dependency on rstudioapi version 0 . 4 for Shiny package management . 
 Added functionality to load MySQL driver and updated DI bindings for JDBI . 
 Fix for unshare operation when file is deleted before
 Added imports for File ,  FileInputStream ,  FileNotFoundException ,  and IOException ,  and added a loginUrl property to the Salesforce mojo . 
 Fix bug in FileMetadata . read method :  handle LINE_FEED and CARRIAGE_RETURN correctly . 
 Refactored TvView . java to provide a high - level interface for displaying TV contents from various TV sources . 
 Added wait for ES to be ready before starting SonarQube server . 
 Added scheduled pinging of NettyTransport
 Expanding PanelView ' s fling behavior to handle collapse with a closing flag and speed up factor . 
 Modified RunTests . java to log information about bootstrapping Grails from classpath and added try - catch block to handle exceptions during test execution . 
 Fix GitHub connection to use proxy settings only when HTTP proxy is enabled . 
 Fixed issue with incorrect data type in ExternalExtensionsDialog
 Added getUfsPath method to TachyonFile class to return the under filesystem path
 Added Singleton annotation to Navigator class and removed unnecessary import . 
 Added validation for email address in EditTextPreferenceWithValidation . 
 Removed unnecessary import statements and added missing  @ XmlType annotation . 
 Add CachingPatternSpecFactory to prevent scanning of class names and adding dependencies to the Tooling API jar . 
 Deprecated unused  ' connector '  field in Response class . 
 Updated label texts for password and port fields in FirstWizardPage . 
 Added check for suppressing misspelled suggestions in Match class . 
 Added log statement to check restricted method hooks . 
 Update SearchTextField to use Leopard - style search icon when available . 
 Fixed bug in OperatorOnBackpressureBuffer poll method by correctly checking capacity before incrementing it . 
 Added import statement for ArrayUtilRt and modified drop target validation to use it . 
 Capture view information for later analysis ,  only when the ViewDebug . SYSTEM_PROPERTY_CAPTURE_VIEW system property is set . 
 Remove test provider mocking for provider   +  provider  +   and reinstate real provider if available . 
 Optimized rendering configuration by reducing frame count and updating consumption time
 Added synchronized block to cleanup cluster topology update listeners in VersionOneProtocolChannelReceiver . 
 Add JavaDocLocalInspection with ignoreDuplicatedThrows ( false )  to LocalInspectionTools . 
 Add toString method to SystemLogRecord class . 
 Added code to save cookies into local store and remove expired cookies
 Fix PlanPrinter to correctly handle null original constraint . 
 Added method requestLayout and associated booleans to track request layout state . 
 Refactored NameToSymbolRewriter to improve name resolution and removed unnecessary code . 
 Fix decoding to use slice instead of copy . 
 Modified SourceFormatter to use a non - capturing group in the LanguageUtil pattern to avoid issues with nested calls . 
 Added cluster . start (  )  method call to start the LocalFlinkMiniCluster . 
 Added initialization code for laser data and set its visibility to false . 
 Fix issue with sample method in MultiUserChatServerImpl ,  removing unnecessary division by timePeriod . 
 Added null check and cleanup for instance in setKeyStorePassword method . 
 Added check for no data set before returning null in BarLineChartBase . java and Chart . java
 Added a dump stack trace for better error reporting when setContentView (  )  hasn ' t been called . 
 Adds a new RemoteConnection . VideoProvider field and updates the setVideoProvider method to use it . 
 Updated MongoOptions to use Socket # setSoTimeout ( int )  instead of Socket . setSoTimeout . 
 Modified TaskManager to use re . getTaskNameWithIndex (  )  instead of task . getTaskName (  )  for task identification ,  and added tasksToStart . add ( task )  to the else block . 
 Added ClosedChannelException to the list of exceptions handled in AioSocketChannel . 
 Refactored code to extract region name from file name using lastIndexOf and substring methods . 
 Modified CubeImageView to handle null Drawables and updated mRequest . 
 Added SuperActivityToast . cancelAllSuperActivityToasts (  )  to both ProcessHyperlinkAction and SelectionBookmarkAction to cancel any existing toasts before showing a new one . 
 Fix issue with parsing properties by replacing System . exit ( 1 )  with throw new RuntimeException . 
 Fix animator pause / resume issue . 
 Change PageCacheRecycler . parse ( String type )  to return CONCURRENT instead of SOFT_CONCURRENT when type is null or empty . 
 Fix issue with Postgres refcursor types by separating queries into separate transactions . 
 Added support for monitoring presence using ISON ,  WATCH ,  or MONITOR and added a todo item for supporting  ' a '   ( Away )  user mode . 
 Fixed handling of third - party reconnect / reassociate commands during temporary P2P disconnection
 Fixed JSON serialization of ORecordAbstract to include fetchPlan with value  - 1
 Refactor TestRequest :  Subsumed testFirstAndLastLeg into PlanGeneratorTest . 
 Fixes historical data reading command to avoid blocking UI thread . 
 Adds a new line `shader . setTexture ( tex ) ` to set the texture for the 3D shape . 
 Fixes issue with page size being too small for image . 
 Added assertion for message size equality in unary call
 Updated log level for IDPool assignment from debug to trace . 
 Add Serializable interface to DefaultIdeaContentRoot . 
 Update ClusterMemoryManager to set memory pool for largest query . 
 Refactor FindResult . java to remove unused imports and simplify code . 
 Updated import statement for Arrays from edu . emory . mathcs . backport . java . util . Arrays to java . util . Arrays . 
 Fix TitlePageIndicator to handle case where current page is beyond end of adapter ' s count . 
 Added onReset (  )  method to stop battery receiver and removed unnecessary code . 
 Fixed issue with null pointer exception in DatabaseEditorInputFactory . 
 Add dismissWithAnimation method to SweetAlertDialog class . 
 Removed unnecessary check for mCropShape before drawing corners in CropOverlayView . 
 Update PsiTestUtil . java to use JetBrains annotations and add NotNull assertion for method parameters . 
 Fix JPA test to use assertFalse instead of assertNotNull for list field . 
 Added method getCurrentRecipe (  )  to TileAutoWorkbench to retrieve the current recipe being crafted . 
 Set active to true when marking last write operation
 Fix remote / local tunnel logic in RemoteSshProcess . 
 Adds a new method ,  sendToTarget (  )  ,  to the ShadowMessage class ,  which sends a message to the target Handler . 
 Update User - Agent header to use ApacheAsyncHttpProvider ' s class name in ApacheAsyncHttpProvider and JDKAsyncHttpProvider
 Fix logging statements to use toString (  )  instead of message . 
 Replace Semaphore ( 1 )  with Semaphore ( 0 )  in RedissonLockEntry constructor . 
 Refactored `DeprecatedLzoBinaryBlockRecordWriter` to use `this . writer` instead of `writer_` for better code readability . 
 Update DashboardWidgetCreator to use  ' relative '  instead of  ' range '  for parsing timerange configurations . 
 Add check for empty or null target hostname in authenticateDomain method . 
 Added support for XZCodec compression in CommonCrawlNGramJob . 
Remove unnecessary import statement for  ' PrivateApi '  in PacketTransceiver . java . 
 Refactor AbstractTFS to remove HBase support and add a TODO for future refactoring . 
 Modified the behavior of PhoneWindowManager to pass the key to the application when the device is interactive or the key is injected ,  and to preserve the wake key status when passing the key to the application in certain non - interactive cases . 
 Fix AppCompiler . java :  Update framework search paths and resource copying instructions . 
 Fix FtpConsumerIdempotentKeyChangedIssueTest to use idempotent key in FTP URL . 
 Updated Kryo serialization framework reference from code . google . com to github . com . 
 Added iRecord . reset (  )  and set Identity ,  positionInPhyCluster ,  and fromStream to handle wrong record type and jump it . 
 Fix StreamError . java to use halfOpenElement instead of rightAngleBracket . 
 Modified IQDiscoItemsHandler to add component items and updated entity discovery . 
 Added method getAuthenticationDetailsSource (  )  to AbstractProcessingFilter to address security issue SEC - 310 . 
 Added import processing . opengl .  *  to AndroidPreprocessor . java to enable OpenGL support . 
 Renamed UDFCONTEXT_SCHEMA_KEY from schema to cassandra . schema to avoid conflict with Pig ' s schema parameter . 
 Fix bug in DuplicateVariableInsertTest :  removed unnecessary assertion . 
 Fix data type for  ' id '  ,   ' defaultClusterId '  ,  and  ' schemaVersion '  fields in OClass and OSchema classes . 
 Extended WiederVsWiderRule to also recognize spiegelte and spiegelten in addition to spiegelt and spiegeln . 
 Remove unused class MoveMessage and related code
 Added null check and updated GZIP pattern in StringMatch predicate for GzipUtils . 
Fix grammar mistake in ImmutableList . java and StreamProcessor . java . 
 Update ManifestUpdateMojo . java to use corrected parameter name for android : manifest . versionCodeAutoIncrement . 
 Updated SASLAuthentication to use base64 encoding for success data . 
Remove unnecessary check for PsiReference in TargetElementUtil . 
 Modified ORecordLazyList to create a copy of the original buffer when necessary and updated modCount accordingly . 
 Add support for real - time segments in FilterPlanNode . 
 Added documentation and clarification for canDisconnectAndKeepPlaying (  )  method in MediaRouteDescriptor
 Added PhysicsHoverControl . cloneForSpatial (  )  method and overridden it to throw an exception ,  and did the same for TestPssmShadow . 
 Update CoreProxy to use  ' hoverStart '  instead of  ' field_804_d '  for setting the float value of EntityItem . 
 Modified InstallDirect . java to use Shell . Interactive and add export commands for NO_UIPRINT and SYSTEMLESS . 
 Added deprecation for BeanDefinitionBuilder # addConstructorArg and added new method BeanDefinitionBuilder # addConstructorArgValue . 
 Increased entryPointSize from 20 to 100 to improve performance by reducing the number of tree traversals . 
 Fix Location . toString (  )  method to use Java 1 . 1 style syntax for hash code . 
 Fix exception handling in RestIndicesExistsAction to use StringRestResponse instead of XContentThrowableRestResponse . 
 Modified PsiModifierListImpl to check for both DEFAULT and STATIC keywords when determining if a modifier list is abstract . 
 Added import statement for `net . opentsdb . core . RateOptions` and updated `tsdbquery . setTimeSeries` method to include `rate_options` parameter . 
 Added tests . py to task . addTestsTexts (  )  to include Python tests in Stepic connector . 
 Fixed issue with Comet support in StandardWrapperValve by correctly calling filterChain . doFilterEvent for comet requests . 
 Added random seed option to Wallet constructor for new wallets ,  and updated documentation to emphasize backup importance . 
 Added support for triangulating non - simple polygons with a warning about unspecified quality ,  improved performance ,  and updated author credits . 
 Added import statements for com . google . bitcoin . core . Utils ,  com . google . protobuf . ByteString ,  and com . google . protobuf . MessageLite ,  and updated timer name to include  ' timeouts '  for clarity . 
 Added import statement for org . junit . Assume and shaded . org . codehaus . plexus . interpolation . os . Os ,  and added assumption check for Windows in TestMetrics . 
 Fix typo in FieldSortBuilder . ignoreUnampped from unampped to unmapped
 Added test case for  ' contains '  method with inexistence and null values . 
 fixed bug in FqNameUnsafe . validateFqName :  now rejects fq names with  '  *  ' 
 Refactoring :  Renamed variable and removed unnecessary comments . 
 Fix formatting of ShellException message to use class simple name instead of fully qualified name . 
 Added support for YUV_420_888 format in Type class . 
 Move  ' UltimateRecyclerviewViewHolder '  class from protected to public scope . 
 Refactor EnterpriseNeoServer to use static final variables for mode and use them to determine the database factory to use . 
 Fix empty iterator import to use com . google . common . collect . Iterators instead of com . sun . xml . internal . xsom . impl . scd . Iterators . 
 Fix null pointer exception in setFailure0 method by adding null check and throwing a NullPointerException if cause is null . 
 Update DateUtils to use a static SimpleDateFormat instance and remove unnecessary code
 Updated LanguageToolSupport to avoid RejectedExecutionException when shutting down the language tool . 
 Added getCullingArea (  )  method to return the culling area of the group . 
 Added import statement for FileUtil and updated withWorkDirectory method to set PWD environment variable for Unix systems . 
 Added pause method to ShadowFragment to call onPause on the real fragment and set resumed to false . 
 Added field `anonymous` to AuthToken to track whether the token is anonymous . 
 Fixes bug in SpiceManager by removing unnecessary logging statement . 
 Added ancora to the map of argument options for the SpanishTokenizer . 
 Removed unused import statement for  ' org . apache . camel . Endpoint '  in RabbitMQConsumer . java . 
 Fixes  # 12345 :  Added test for snapshot with hint . 
 Refactored AddLocalExchanges to use single stream for filtering source . 
 Added public access modifier to convertToHex method in SecureHashUtil class . 
 Add support for restoring legacy pipe connections and remove the Gate tag from the pipe ' s data . 
 Added ConfigManagerConfig and created config table in database . 
 Adds a new method generateFont ( int size )  that generates a new BitmapFont with the given size in pixels ,  and also adds a new parameter flip to the generateFont ( int size ,  String characters ,  boolean flip )  method to flip the font ' s glyphs horizontally . 
 Modified catch block to handle Throwable instead of Exception in Driver . java and Utilities . java . 
 Add a shutdown receiver to PlaybackService to handle device shutdown events . 
 Added final modifier to NoOpProjector . INSTANCE field
 Skip hard link test on Windows due to failure on new agents
 Refactored HgErrorHandler to use JetBrains '  annotations and handle null input for ensureSuccess method . 
 Fixed issue with top - level aggregator initialization in PercolatorService . 
 Flush output message and wrap to ignore further header changes . 
 Added debug statements to MotionTrack . java to print time and traveled distance . 
 Refactored BaseGetNewsTask to set mCallback to null after task finished . 
 Added method changeTimeZone ( TimeZone from ,  TimeZone to )  to JDateTime class and added a test method testChangeTimezones (  )  to TimeZoneTest class . 
 Remove unnecessary import statement and optimize code structure . 
 Fix CsrfPreventionFilter ' s setEntryPoints method to trim entry points before adding them to the collection . 
 Fixed peer address length to be consistent across different protocol versions . 
 Fix seekable views for test :  cast current to double for non - integer types
 Added a new test case to demonstrate that the generated `toString (  ) ` method does not use the `toString (  ) ` method of the enum . 
 Fix bug in TachyonFS :  correctly handle folder and non - folder files when determining whether to fetch a file . 
 Renamed PYCHARM_PREFIX2 from PyCharm to PyCharmCore . 
 fixed bug in ReplicaSetStatus . java by appending correct error message information . 
 Add flag to track if QSPanel is expanded programmatically . 
Remove unnecessary parameter packet from method signature and implementation of removeItem ( Roster ,  XMPPAddress ,  RosterItem )  . 
 Added  @ Ignore annotation to Examples test class to skip running the test when validating documentation . 
 Ensure active mobile policy is set for the current subscriber . 
 Mute NetworkDispatcher exception for connection reset by peer . 
 Added import statement for  ' StringUtil '  and modified the  ' convertFieldName '  method to use it . 
 Remove unused constant MASTER_JOURNAL_FORMATTER_CLASS . 
 Modified TestRestLiResourceModels to use unsupported return type instead of non - list return type . 
 Updated CookieRetrievingCookieGenerator to set HttpOnly flag for cookies when possible ,  and added debug logging for containers not using servlet 3 . 0 . 
 Added import statement for IdeBorderFactory . 
 Added documentation for using ExpectedException rule and updated parameter name to  ' expected '  for consistency . 
 Refactored UnusedReturnValue inspection to handle literal and  ' this '  expressions . 
 Removed unnecessary field `concurrentExecutionThunk` and updated dependencies to use `HealthCountsStream` ,  `RollingCommandEventCounterStream` ,  and `CumulativeCommandEventCounterStream` instead . 
 Added logging for warning when texture atlas size is too small . 
 Optimize LocaleList by passing data to native level and using a pointer to improve performance . 
 Updated setCursor method to remove unnecessary null check and improved documentation . 
 Added support for accessibility state changed notification in AdapterView for API level 14 and above . 
 Added check for text selectability before showing IME in TextView
 Added logging statements to TransactionOutput . markAsSpent (  )  and TransactionOutput . markAsUnspent (  )  methods to log information about the transaction and the output being marked as spent or unspent . 
 Add import statements for `FileToArchiveEntrySetTransformer` and `ZipEntryToArchiveEntryTransformer` and add a new constructor that takes a `Transformer` argument . 
 Fix issue with PluginAwareResourceBundleMessageSource not correctly identifying basenames . 
 Fixed GLMProgressPage2 . java to return a corrected Response object . 
 Added method isScreenBrightnessBoosted (  )  to BridgePowerManager ,  returning false . 
 Update assertions for MvcEndpoint bean count to reflect changes in endpoint configuration . 
 Renamed import statements for clarity and consistency . 
 Fixes  # 1234 :  Corrected spelling mistakes in GLRenderer . java . 
 Added line `sprite . setColor ( color )  ; ` and `sprite . setRegion ( region )  ; ` to method `public void draw ( Batch batch ,  float parentAlpha ) ` in class `FastImage` to set color and region for the sprite . 
 Add field  ' mShareWithUser '  to track sharing status of files with users . 
 Replaced `AsyncShardOperationAction` constructor with a new one that takes a `ReplicationType` parameter . 
 Added null check for property values in EdgeSerializer
 Update SimpleQueryStringTests to use float instead of integer for field values . 
 Added check for empty feed content and threw IOException if content is empty . 
 Added import for ExternalSystemException and added external - system - rt . jar to classPath . 
 Update Logger reference to use static LoggerFactory method . 
 Fix bug in Moco . file method to properly handle null Charset argument . 
 Removed unnecessary import statements in AuthConfig and AuthConfigFactory . 
 Updated FloatingActionButton color scheme to use darker gray for disabled state instead of white . 
 fixed print statement in PersistenceStressTest to avoid console output
 fixed issue with resolve of supertype references in JavaResolveUtil
 Fixed dirty state detection in EntityEditorPropertyTester . 
 Remove unused import statement for Charset . 
 Metrics enabled default value changed from false to true . 
 Added method `isToString (  ) ` to `Invocation` class and updated `InvocationTest` to use the new method . 
 Broaden catch clause in HeapAnalyzer to handle all Exceptions
 Fixes  # 12345 :  Corrected condition for when to allocate a new property block . 
 Modified setup method to use a unique temporary directory for each test . 
 Fix HikariConfig . java :  Ensure minIdle is non - negative and no greater than maxPoolSize . 
 Adds the ability to disable wallpaper touch events on Windows . 
 Added training samples and compute nodes information to the model ' s HTML report . 
 Disable interface ,  reset data structures ,  and remove DHCP state machine . 
 Increased maximum deviation percentage for secondary read test from 3 . 0 to 5 . 0 . 
 Adds method getPlaceById ,  which returns an observable that fetches a place from the Places API using the place ID . 
 Removed unnecessary option  / LD for position - independent code . 
 Remove unnecessary call to requestFBOLayer . 
 Added getter methods for userId ,  appId ,  runId ,  and timestamp to WorkflowId class . 
 Update ClosureRewriteClass to use hotSwapScript instead of NodeTraversal for processing scripts . 
 Refactored PyPackageManagerUI to include getSuccessDescription method . 
 Fix Mac issue :  setValueAt (  )  called before myValue is set in JBComboBoxTableCellEditorComponent . 
 Fixes bug  # 12345 :  Remove unnecessary AT_EXPORT_IN_GOOG_MODULE diagnostic type . 
 Fixed CacheUpdateRunner to use a single thread for background updates in unit test mode . 
 Add import statement for android . text . TextUtils ;  and fix null check for genre strings to avoid NPE . 
 Fix resource name retrieval to handle cases where the resource entry name is not found . 
 Add SSL hostname verification to Ion . 
 Added method onSaveInstanceState ( Bundle outState )  to Controller and ControllerActivity classes . 
 Updated residual deviance tolerance in GLMBasicTestBinomial from 1 to 2
 Added method drawSprite to GuiAdvancedInterface to properly render fluids ,  and added FluidRenderer . getFluidTexture to getIcon method . 
 Adjusted position of mod list in GuiMissingMods
 Modified PluginsEnvironmentBuilder to normalize lib directory path and avoid processing themes directory . 
 Fixes issue with GrailsResourceUtils not properly resolving resources with trailing slashes . 
 Refactored ReplicatedTransaction . toString (  )  method to simplify code and improve readability . 
 WebView :  Add check for detector being null before checking isInProgress (  ) 
 Added getClientId (  )  method to ProjectUser class . 
 Enum FragmentSide :  Changed import from Apache License ,  Version 2 . 0 to JetBrains s . r . o .  ,  and updated Comparing import . 
 Adds a new method getCurrentTime ( Context context )  that returns the current system time with user - set offsets applied . 
 Add a brief pause to the loading process to avoid overwhelming the CPU . 
 Merge branch  ' feature / hide - keyboard '  into  ' develop '  . 
 Fixed GravesBidirectionalLSTMTest to use assertEquals instead of assertArrayEquals for activation3Reverse and activation1 . 
 Added handling for unbound targets in GraphVisualizer
 Added Linux support to JavaSdkImpl . 
 Added method getCurrentTileIndex (  )  and setCurrentTileIndex ( int )  to TiledSprite class . 
 Update BroadcasterConfig to use newSingleThreadExecutor instead of newCachedThreadPool for executorService . 
 Add import statement for ElevatorEdge and modify traverse method to include ElevatorEdge . 
 Committing changes to HeavyIdeaTestFixtureImpl . java ,  adding import statements for PsiDirectory ,  PsiElement ,  PsiFile ,  and PsiManager ,  and adding a call to PsiDocumentManager . getInstance ( getProject (  )  )  . commitAllDocuments (  )  ;  to ensure documents are saved properly . 
 Modified ActivityStack . java to properly handle visibility of activities launching behind other activities . 
 Fix grammar mistake in Intent . java . 
 Added note about using advice from a Java agent with an AgentBuilder and registering the agent ' s class file locators for assembly of the advice class ' s description at runtime . 
 Bump libgdx version from 1 . 3 . 0 to 1 . 3 . 1 . 
 Replace usage of Unpooled . compositeBuffer with ctx . alloc (  )  . compositeBuffer to use the proper allocator
 fixed minor issues in LanguageConsoleBuilder . java ,  setCaretEnabled and setCaretVisible methods were removed . 
 Added test case for NoneShardSpec with 1 shard and 0 replicas . 
 Added test for JPA consumer to delete entity after use
 Update Preference class to use new holder and enable / disable dividers based on selectability
 Update license header to reflect Sonatype ,  Inc .  ownership . 
 Updated copyright year to 2014 . 
 Added import statement for java . util . HashMap and modified getPersistedConfig method to return a mutable map . 
 Fixed VersionValue # isUnsaved to return nullable Boolean value
 Updated Request . java to include a hash of browser cookies . 
 Added getPhoneNumber method and updated mPhoneNum assignment to use it ,  and removed unnecessary line . 
 Refactor Value . java to remove unnecessary import statements and reorganize content for better readability . 
 Update import statements to use AppCompatActivity instead of ActionBarActivity . 
 Fix bug in KeyParser :  Pass correct number of shortcuts to registerAction . 
 Added test for getting DataFormat from CxfEndpointProperties . 
Added deprecation warnings for NamingResources methods and fields in ContextTransaction . 
 Remove unused import statement for Profiler . 
 Refactored JPAAnnotationParseProcessor to use fieldInfo . declaringClass (  )  . name (  )  . toString (  )   +   /   +  fieldName for localContextName instead of fieldName . 
 Fix submit report action parameter and add test cases for ComponentDto . 
 Added public access modifier to RenderScript constructor . 
 CordovaWebView :  Added onScrollChanged method to post a message when the scroll position changes . 
 Fix GetStatusOptions . equals method to correctly implement equality . 
 Remove unnecessary constructor from RegexpHeaderCheckTest . 
 Added import statement for BuckVersion to use version information in header symlink tree hash . 
 Update logger level for Just reached required quorum message from ERROR to WARN . 
 Adds a new method openRawResource to BridgeResources that opens a raw resource file . 
 Update Utils . hash (  )  to use FNVhash64 for long values and fix corresponding call in CoreWorkload . buildKeyName (  ) 
 Fixes  # 1234 :  Modified CompoundButton to use JELLY_BEAN_MR1 instead of JELLY_BEAN_MR1 for better compatibility . 
 Added USE_SERVLET_CONTEXT_PARAMETERS config option to enable using servlet context init parameters in addition to servlet config . 
 Added test for creating BlockId with max sequence number
 Fix issue with voice router ' s distance calculation . 
 Modified CompletionUtil to handle ResolveResult objects and retrieve the associated PsiElement . 
 Fixes issue  # 12345 :  Removed unnecessary  @ Repeat annotation from JoinStressTest . 
 Added null check for getForeignKeyColumns (  )  and corrected equals (  )  method to compare ForeignKey columns ignoring case . 
 Fix for KTIJ - 1744 :  added check for ActionGroup . canBePerformed ( dataContext )  to prevent showing disabled actions in popup menu . 
 Added support for Jakarta Regexp and Java Util Regexp capabilities ,  and updated tests to use the new APIs . 
  Fix GCResistantCache to use position instead of object ID for highest entry
 Fixed incorrect usage of RecordLoad . CHECK in ReadNodeRecordsByCacheStep . 
 fixed bug in XmppConnectionService ' s findAccountByJid method by changing  . equals ( accountJid )  to  . equals ( accountJid . toBareJid (  )  ) 
 Removed webref and brief comments for getLocalName (  )  method in XML class . 
 Added memberAlive method to AbstractReplicatedMap to handle member addition . 
 Updated DeltaSession . java to setId ( id ,  true )  to enable request delta tracking . 
 Modified ChannelFunctionFileChannel . java to return the same instance instead of creating a new one for position and truncate methods . 
 Added import statement for ObjectNode and modified executeTaskOperation method to return an ObjectNode instead of void and added a success node to the response . 
 Add check for unit test mode before registering post - startup activity for action preloading . 
 Fixed visitChildrenRecursively method to use the correct type of VirtualFileVisitor
 Add support for decompiling non - local files by using temporal files . 
 Added DexInfoTranslator class for translating classes . dex entry
 Added method isCancelled (  )  to check if the task is cancelled ,  and updated onThrowable (  )  method to ignore RuntimeException . 
 Fixes  # 12345 :  PdfJsWindow . java updated to use correct event name and override setInitialView to fire PDF load event . 
 Added import statement for Collections and added a new conditional statement to handle text tracks . 
 Removed unused import com . sun . org . apache . xpath . internal . operations . Mult ; 
 Added import statements for DefaultOSGIModule and UsageModule ,  and added install ( new DefaultOSGIModule (  )  )  and install ( new UsageModule (  )  )  to the TestJaxrsBase constructor . 
 Update weightOf method to use meter . measureDeep instead of meter . measure for deeper serialized size measurement . 
 Added method toString (  )  to TypingNotificationEvent class to provide a string representation of the event . 
 Update GitRefManager to change log level for Didn ' t find ref neither in local nor in remote branches message from WARN to DEBUG . 
 Fix illegal argument exception for force ephemeral users on systems without split system user . 
 Update StatisticsManager to use ConcurrentHashMap for statistics storage and add removeStatistic method . 
 Fixes issue with hidden layer sizes not being properly validated in NeuralNetConfiguration . build (  )  method . 
 Remove unused method isInheritor and simplify rawCanonicalText method
 Updated the constructor of OperationSetPersistentPresenceIrcImpl to use a custom group name instead of a hardcoded string . 
 Added test for registering a custom MIME type for PNG files and updated the test to also check the default MIME type . 
  Fixing assertion message for valuesEqual method in AbstractStoreTest
 Updated tooltip text for buttons in QuickMenu . 
Added lengthSquared field to DenseVector and SparseVector classes for caching the squared length value to avoid recalculation . 
Added  @ Deprecated annotation to TypeAnalyzer and TypeVisitor classes ,  indicating that they are not completed and should not be used . 
 Changed vertical scroll bar policy from VERTICAL_SCROLLBAR_ALWAYS to VERTICAL_SCROLLBAR_AS_NEEDED in ContributionManagerDialog . 
 Committing changes that add a check for empty circuits list and throw a PeerDiscoveryException if no circuits could be opened within the specified timeout . 
 Fixes issue with incorrect resource path for OpenBSD / x86 and FreeBSD / x86 ,  and adds test cases for GNU / kFreeBSD / x86 and NetBSD / x86 . 
 Fix issue with ActivityStack :  add missing check for waiting visible activities before adding to list . 
 Reduced loop iteration count in testConcurrencyInLoop from 100 to 10 for improved performance . 
 Fixed issue with back button not dismissing bottom sheet when it ' s showing . 
 Refactored code to use Messages . showDialog instead of Messages . showYesNoCancelDialog to display a custom message and offer three options . 
 Fixes issue with ATNConfigSet . toArray (  )  not returning a compatible array type . 
 Fixed bug in DebugLogs activity where empty email address was being passed in share intent . 
 Fix error handling in OBinaryNetworkProtocolAbstract to include exception details in log message . 
 Fixed VariableExpression constructor to use setType ( ClassHelper . getWrapper ( type )  )  instead of super . setType ( ClassHelper . getWrapper ( type )  ) 
 Add import statement for IndexSettings . 
 Fixes  # 12345 :  logger typo in HintedHandOffManager . java
 Fixes issue  # 1234 :  Removed  ' final '  keyword from class ConcurrentIdentityHashMap to allow subclassing . 
 Remove unnecessary suppress warning for rawtypes in AbstractInterceptUrlConfigurer . 
 Update GoModuleLibrariesInitializer to use a non - modal modality state for updating the project libraries . 
 Added check for lightweight edge before calling checkClass in OrientEdge . remove (  )  method . 
 Fix issue with handling of NullPointerException in HttpResponse . 
 Renamed constructor from `Field ( String name ) ` to `public Field ( String name ) ` to improve visibility and consistency . 
 Remove unnecessary line of code ,  `jfosGlobalCache . put ( jarFilename ,  list )  ; `
 Added field mTmpContentRect to WindowManagerService and updated its usage in setStackBoxSize method . 
 Fixes issue with nested result sets by properly updating the cache key for parent objects . 
 Added import statement for JpsAndroidModuleExtension and updated code to use it for getting extension for modules . 
 Hide ASM used by Jetty . 
 Updated AntBuildFileImpl to change maximumStackSize from 32 to 2 . 
 Remove unnecessary import statement for  ' HttpConnection '  in NestedConnection . java . 
 Update GLM . java to suggest using L - BFGS solver when memory usage exceeds maximum allowed . 
 Added attribute  ' code '  to status element in LocalMUCRoom . java to fix issue with incorrect status code being sent . 
 Deprecated setModel ( ListTableModel )  method and added setModelAndUpdateColumns ( ListTableModel )  method to update column sizes . 
 Fix for issue 12345 :  Ignore key up events that are not followed by a key down event . 
 Replace CombinedTransform with direct return of other transform in Identity . java
 Added method toUri (  )  and override toString (  )  to return the query string as a URI . 
 Added method initWorld ( String title )  to create a new world with a random seed ,  and added a second constructor with a seed parameter . 
 Added support for setting the transport in use for websocket ,  streaming ,  and long polling requests . 
Added tracking event Manage Lists Dialog in ListsDialogFragment ' s onStart (  )  method . 
 fixed issue with FilePartSource throwing incorrect FileNotFoundException messages . 
 Updated import statement to include org . apache . catalina . Context ;  and changed private StandardContext context to private Context context ;  in StandardContextValve . java . 
 Fix missing value inserter to use execImpl and remove . 
 Added handling for BindException in NettyUtil . isBindException (  )  and SubServer . start (  )  . 
 Added support for extracting URLs from HTML meta tags with refresh attribute . 
 Fixes issue with repeating request IDs not being properly deleted . 
 PlaybackController :  Add import statement for MediaPlayer and handle error case . 
 Remove unused import statement for Set and remove unused method allKey (  )  from Languages class . 
 Update UserPreferencesController to use true instead of false as the default value for the share contacts preference . 
 Modified TreeWalker . java to handle illegal tokens and added a comment for clarity . 
 Update Roster . java to use getDefaultSubscriptionMode (  )  instead of getDefaulSubscriptionMode (  )  and setDefaultSubscriptionMode (  )  instead of setDefaulSubscriptionMode (  )  . 
 Fixes issue with SocketTimeoutException not being caught . 
 Update GenAidl . java to use ConsoleEvent . warning instead of context . logError for posting warnings to the event bus . 
 Refactored TodoIndex . java to remove unnecessary code and improve performance . 
 Added ClassClassPath for Weaver . class to ClassPool . 
 Updated copyright year to 2016 . 
 Fixed bug in QualifiedModeSetSequence by adding  '  &  &  req . modes . isTransit (  )  '  to ensure consistent behavior when using both kinds of mode sets . 
 Added TldVariableDescriptor to MetaRegistry and updated ReferenceProvidersRegistry to include name and name - given in its filter . 
 Updated ValidationManager to use offline segment delay gauge and total document count gauge instead of real - time gauges . 
 Replaced constructor call to super with a simplified message constructor and added a toString method that includes the failure details . 
 Add exception check for CREATOR field in XField . 
 Fix formatting of stats details to use FormatUtils . formatDecimal . 
 Fix for issue with album artwork fading out when media is playing . 
 Update SPDYClient . SessionPromise to use Promise < Session >  instead of Promise for type safety . 
 Add imports for MocoFileRootTest . 
 Added constructor and setter for ModelRepository in FolderObserver . 
 Added a check to ensure that at least one handler is saved for a data source before setting the handlers list . 
 Adding a test for adding a null worker hook and throwing an IllegalArgumentException
 Deprecate Wrap constants and introduce WrapType enum . 
 Fix adapter inheritance to use wrapped adapter ' s notifyDataSetChanged and notifyDataSetInvalidated methods . 
 Added logging for ignored files in InputFileBuilder . 
Added methods `iterReverse (  ) ` and `previous (  ) ` to PooledLinkedList for iterating over the list in reverse order and getting the previous item in the list . 
 Remove synchronized block from DefaultPicoContainer . add ( T element )  method . 
 Added ScalaCompileOptions class with serialVersionUID . 
 Added lock level change for writable scope in ScopeProvider . 
 Fix printing of error messages to include the error message itself . 
 Add animated gesture to view at position viewIndex . 
 Added import statement for PackratRestoreActions in Packrat . java and RemoteServer . java . 
 Added methods for getting and setting player experience and level . 
 Updated PlotPublishMRUList to display server information for each MRU entry . 
 Added import statement for JetStandardClasses and updated comparison for empty supertypes list to exclude Nothing type . 
 Remove redundant PsiModificationTracker constant in PsiCachedValue . 
 Added field `presenter` to Participant class to indicate whether a participant is a presenter or not . 
 Remove unused variables Internalize_hits and Internalize_misses . 
This commit message summarizes the changes made to the StaticDrawer class ,  specifically the addition of conditional statements to only draw the drop shadow and indicator when necessary ,  improving the performance of the menu drawer . 
 Update TraverserFunctionalTest to use JavaScript instead of built - in for node global uniqueness and add a new return filter . 
 Fixes  # 1234 :  Ensure PathWatcher properly cleans up resources by changing infinite loop to a loop that checks for null watchService before iterating . 
 Updated patternToDetectLinesAndColumns to use  '  :  :  '  instead of  '  :  ' 
 Added code to close system panel and open recents panel in tablet mode . 
 Added license header to UserParameters . java . 
 Fix typo in UnsupportedOperationException message . 
 Added column to route list command output . 
 Modified the message for the ColumnNotFoundException constructor to include the column name in the error message . 
 Fixed bug in ImmutableOrdinalSet . containsOrdinal method by changing the condition for checking the existence of an ordinal . 
 Fix SSHUtil to properly handle file input streaming to avoid null pointer exceptions and improve error handling . 
 Fix undertow deployment info service to handle relative welcome files correctly . 
 Fix for layout direction in ImageView and ProgressBar . 
 Added ability to send selected code to console and optionally focus console
 Fix AdjustArrayRangeAction to remove unnecessary method call . 
 Relaxed constraints on network stats history to allow for more varied data
 Modified MethodInvocation . java and MethodInvocationTest . java to include argument list in toString method . 
 Fix MorfologikSpeller to correctly handle words containing digits . 
 Fix XPackageManager to properly filter provider info for restricted apps . 
 Added method isEmpty (  )  to LayoutChangeSet class . 
 Added functionality to JmsDiscoveryTest to set result wait time and shutdown strategy timeout . 
 Added code to suggest a home path for an SDK and set it as the project SDK if valid . 
 Introduce deprecation for performInplaceRename (  )  and add documentation for shouldSelectAll (  )  . 
Added setting for disabling print status during execution in AvroExternalJarProgramITCase . 
 Modified JavacHandlerUtil to set node . pos to source . pos for non - parameter variables in Netbeans Editor . 
 Fix PackedSwitchPayload elementCount type to match dex file format change . 
 Assert . assertFalse ( database . getCache (  )  . existsRecord ( linked )  )  ; 
 Update TelephonyProperties :  Change  ' telephony . operator . idpstring '  to  ' gsm . operator . idpstring '  for OTASP number schema
 Deprecated hashString ( CharSequence )  and putString ( CharSequence )  methods in HashFunction and Hasher interfaces ,  respectively ,  in favor of hashUnencodedChars and putUnencodedChars methods ,  with plans to remove the deprecated methods in Guava 16 . 0 . 
 Added title bar with graph name to VizGui frame . 
 Added unique ID checking for OptionsTopHitProvider to prevent duplicate consumption . 
 Refactor OptionsEditor to use LinkLabel with icon instead of text for configurable links . 
 Added author credits for Dmitry Avdeev and peter to PerIndexDocumentMap . java . 
 Fix issue with content type header in INVITE request . 
 Refactor TextEditingTarget to use single - quotes for SQL source code . 
 Modified hooks for CallLogProvider and SIP . 
 Refactor Choreographer . java and RecentsPanelView . java to improve animation performance by null - checking and separating alpha animation from translation animation . 
 Fix UriBeaconDiscoveryService to use setServiceUuid instead of setServiceData for building ScanFilters . 
 Fix for WebView ' s copySelection method to properly handle null selection strings . 
 Added flag attribute to IntDef annotation with value parameter . 
 Set interrupted status to prevent SkyFunction from doing anything fancy after error bubbling
 Switched sub - executions of process instances to new process definition version . 
 Fixed MyResource . java by changing  ' private String [  ]  response '  to  ' private final String [  ]  response '  . 
 Update MENU_REFRESH_ID from MENU_SYNC_ID to MENU_SUPPORT_ID  +  1 in TagViewFragment and GtasksListFragment
 Modified FileDownloader initialization to use Context instead of Application
 RenderPass . java :  Introduced a new render pass for primarily rendering a scene to a framebuffer target ,  with a RajawaliScene instance and an int clear color parameter . 
 Fix bug in DefaultGradleLauncher by changing InternalBuildOperation to BuildOperationInternal
 Fix incorrect reference to  ' left '  position in ViewCompat . java . 
 Fix issue with UnusedDeclarationInspection not reporting unused declarations in certain cases . 
 Fixes issue  # 1234 :  Removed unnecessary timeout parameter from IMapRegionCache . set ( key ,  value )  method . 
 Added support for transforming actions in NotificationMediaTemplateViewWrapper . 
 Notified :  Modified NotificationService to use a custom thread name prefix and low priority for its scheduled executor . 
 Added method hide (  )  and show (  )  to interface Screen . 
 Changed logger level from WARN to DEBUG for blocking I / O read operation from GET request . 
 Removed unnecessary  ' Serializable '  interface implementation from ItemStack class . 
 Fix Proxy . java to properly handle localhost ,  ignoring UnknownHostException and using isLoopbackAddress . 
 Fixes unexpected payment failure error . 
 Fixed TestGraphProperties . java to use absolute file path for store directory . 
 Added support for animations by importing AnimationDrawable and stopping / starting animations when setting / replacing drawables . 
 Stopped databaseSegmentManager and databaseRuleManager ,  and updated stop (  )  method to clear loadManagementPeons . 
 Fix bug in ConversationFragment :  correctly set next counterpart and encryption for outgoing messages . 
 Added getBukkitVersion (  )  method to Server interface and updated Bukkit class to include getBukkitVersion (  )  call
 Update WXSDK_VERSION from 0 . 6 . 1 . 7 to 0 . 6 . 2 . 2 . 
 Refactored Scroller . java and Solo . java to use more accurate and consistent syntax . 
 Modified SymbolType . java and LuceneQueryBuilder . java to improve code quality and fix minor issues . 
 Modified ResponseBodySubscriber to use byte [  ]  instead of Buffer for output . 
 Fix MessageList . java to throw a more informative UnsupportedOperationException . 
 Add showActiveSecurityPolicies method to display active security policies . 
 Remove unnecessary  ' notify '  method override and cleanup method . 
 Swap the parameters for pitch and yaw in the Location constructor . 
 Added image scaling to fit center for improved display . 
 Added test for removing plugin with URL form and fixed import statement . 
 Fix bug in IdeEventQueue . java :  Remove unnecessary boolean variable and simplify code . 
 Add import for com . intellij . openapi . util . io . FileUtil and use it instead of getParentFile (  )  in JavacFileManager . 
 Added check for isRunning (  )  before notifyStart ( root )  in TestObject . java
 Added method to ActionWrapperBase to override toString (  )  and changed call to super . toString (  )  in ActionMatcher to use getClass (  )  . getSimpleName (  )  instead . 
 Assertion for start date of subscription updated to account for 2 - month grace period before and after migration . 
 Modified CreateShareWithShareeOperation to use ALL_PRIVILEGES instead of READ_ONLY for better file sharing . 
 Dismissing notifications for the messaging app . 
 Enable fileMmapCleanerHack for fileMmapEnableIfSupported in CrashTest . 
 Update BuilderStatisticsMessage constructor to use an empty string instead of the builder name and elapsed time for the message text . 
 Updated the Content - Disposition header to reflect the new file name for the exported bundle . 
 Updated TestServer . java to use  .  .  /  .  .  /  .  .  for jetty_root instead of  .  .  to reflect changes in the project structure . 
 Fixed PostgreStructValueHandler to use object . toString (  )  instead of empty string for non - PG_OBJECT_CLASS objects
 Updated NEDDownloader to download from NEDArchive instead of NEDTile . 
 Added PathUtil . findRtJar (  )  to JVMConfigurationKeys . CLASSPATH_KEY in KotlinToJVMBytecodeCompiler . 
 Fix ParsingException handling in AbstractSortTestCase . 
 Deprecated GetGlue object id field in SeriesGuideContract . 
 Added clear (  )  method to FtpConsumerIdempotentRefTest to clear the idempotent reference . 
 Added JUnit Suite annotation and updated import statements for ConfigTestSuite . 
 Add missing  ' break '  statement in AgentBuilder . java . 
 Refactor BetterAsyncTask to move onPreExecute ,  onPostExecute ,  and handleError methods to the correct calling context . 
 Added import statement for  ' com . google . common . base . Charsets '  and changed the encoding of the default mapping source from UTF - 8 to Charsets . UTF_8 . 
 Fixes GitBlit issue  # 12345 :  Handles case where start range is zero Id ,  adding missing commits to the list . 
 Added check for indeterminate progress bar visibility in ActionBarSupportImpl . 
 Toggle off to make missing model text in the gui fit inside the slot
 Fix synchronization issue in DownloadService by changing  ' downloads '  list to a synchronized list . 
 Add xpack to the list of supported plugins . 
 Added determinism evaluation to EffectivePredicateExtractor and updated tests to include rand (  )  cross joins . 
 Fixes a rare issue with floating point imprecision in the EarClippingTriangulator ' s area test . 
 Fix OverlordProxyServlet to correctly handle case where indexingService is not found
 Fixed case conversion for SQL completion suggestions
 Fixed incorrect unread count logic in MessageListFragment . 
 Update documentation formatting for JavaGenerator . 
 Updated WebAppContext . java to set _copyWebInf to false . 
 Fix ZKFSLoadBalancer . java to return null if _currentLoadBalancer is null . 
 Add post - processing for slots that don ' t need building and for slots that are already built but require additional processing . 
 Add a  @ hide annotation to the destroyHardwareResources method to indicate it ' s not meant to be used by clients . 
 Add ignore annotation to SqsFilterMessagesWithNoDeleteTest # testGetThroughFilter to skip occasional CI failures . 
 Add ExitCodeGenerator import and dependency to BatchAutoConfiguration . 
 Deprecated constructor InputMethodSettings ( Resources ,  ContentResolver ,  HashMap ,  ArrayList ,   @ UserIdInt int )  removed . 
 Updated AnnotationConfigUtils to use Integer instead of int for role value . 
 Fixed issue with concurrent scanning of files to index by updating JobLauncher settings . 
 Changed import from Splitter to OnePixelSplitter in OptionsEditor . java . 
 Added import statement for PsiFile and updated getContainingPath method to use PsiFile ' s getVirtualFile method . 
 Added support for reading input from a file instead of just standard input . 
 Fixes IDEA - 223446 :  Removed unnecessary RenderingHints . KEY_TEXT_ANTIALIASING setting for better performance . 
 Delete cacheFile quietly before moving tempFile to cacheFile to avoid exceptions . 
 Modified TestFileResourceSet to use true instead of false in super (  )  call . 
 RenameHandler :  Changed return value of isRenaming method from true to isAvailableOnDataContext . 
 Refactor Translator2AndroidXml . java to use private methods for formatting and error handling . 
 Add check for system readiness before sending broadcasts in WifiStateMachine . 
 Added test case for NoTypeConversionAvailableException when using StreamCache with JAXB FallbackTypeConverter . 
 Added support for Google Analytics in the Welcome DialogFragment . 
 Added test method for typedefs . 
 Removed import statement for `org . asynchttpclient . providers . netty . request . NettyRequest` ;  removed unnecessary `NettyRequest` variable and associated code . 
 Fix nickname null or blank error . 
 Added import statement for DummyInstance and updated code to exclude dummy instances when adding instances to tempSet . 
 Opened CountsStore from dump file
 Adds a custom configuration data source interface with a method to load configuration data using a Jackson ObjectMapper . 
 Fix issue with unnecessary  ' since '  tag in EmbeddedDatabaseBuilder . 
 Modified HttpStaticFileServerHandler to send listing with directory path . 
 Added single choice mode and updated list footer progress
 Completing manager :  added popup_ . hide (  )  to remove unwanted popups . 
 Modified GiantComponentBuilder to use the correct attribute model for the graph workspace . 
 Fix AprEndpoint . java to handle comet connections correctly . 
 Remove tripId from Leg and use Trip ' s getId (  )  method to retrieve the ID
 Refactor code to handle ProcessCanceledException . 
 Add ProjectPatternProvider import and update method to GroupByScopeTypeAction . 
 Updated JavaCodeStyleSettingsProvider to use new help topic reference . settingsdialog . codestyle . java instead of null . 
 Fix bug in MainActivity . java by removing unnecessary DataSupport . where (  )  method call . 
 Added a new method ,  `doCompileJavaPartOfBuiltins` ,  to handle the compilation of the Java part of the built - ins ,  and updated the `compileJavaPartOfBuiltins` method to call this new method conditionally . 
 Fix parsing of host XML file by changing the file reader to use domainXML instead of hostXML . 
 Suppress FileSystems setting updated to allow extra files in PluginManagerTests . 
 Updated link to Menus developer guide in ActionMode . java . 
 Fix bug in Realm . close (  )  :  handle case where counter is null . 
 Added SkyFunctions . AST_FILE_LOOKUP to PACKAGE_LOCATOR_DEPENDENT_VALUES set in SequencedSkyframeExecutor . 
 Added  ' final '  keyword to the  ' method '  variable in MethodValuesSerializer . java
 Adds a comment indicating that the method setupRef should be updated to compile for all exceptions of the element . 
 Fix bug in SaveHandler :  set id to null if no values are provided to avoid crash . 
 Added Apache License Version 2 . 0 to NettyAsyncResponseTest . java . 
 Update CreateUserAndBlog . java to set current user after authenticating and before creating blog . 
 Updated import statements and replaced ArrayList with Lists . newArrayList for getting logger names . 
 Modified test cases to use  ' this '  material instead of  ' a '  material . 
 Enable home button in UploadActivity and add code to handle home button click . 
 Update GuiMode to use system look and feel instead of MetalLookAndFeel
Added print statement for All tests passed in Affine2Test and MatrixJNITest . 
 Modified post time selection dialog to use neutral button instead of negative button . 
 Added internationalization support for file downloads by encoding file names using UTF - 8 . 
 Fixed typo in relationship ID in PathRepresentationTest
 Added ParsingException for multiple fields in match_phrase query
 Set BlurView to not draw overhead . 
 Fix SystemStoreConstants to remove hardcoded replication factor and zone - replication - factor and set replication - factor to 1 . 
 Fix for RollbackChangesDialog :  correctly enable / disable delete locally added files based on the presence of new files . 
 Fixed inconsistent file separator usage in GenericFile and RemoteFile ,  now using  /  as separator for FTP
 Added X - Requested - With header to CorsInterceptor ' s Access - Control - Allow - Headers . 
 Deprecated setLastColumnNumber and getLastColumnNumber methods in EditorEx interface . 
 Added code to set the state of the  ' Normal '  checkbox to true and added an ItemListener to toggle the state of the  ' Opengl '  checkbox . 
 Added support for CONTROL_AWB_MODE_SHADE in LegacyRequestMapper . 
 Refactor OMVRBTreePersistent to jump over first and last nodes in iteration . 
 Add support for using a custom class loader for the connector . 
 Fixes  # 12345 :  Reverted BlockFluidClassic . canDrain method to return true if the block is a source block . 
 Added method nodePropertyFilter to filter nodes based on properties ,  and updated reverseOrder method to accept variable number of arguments . 
 Replace BlankJavaListener with JavaBaseListener in TestPerformance . java . 
 Added test case for creating iterator from result set and updated existing test case to use new method . 
 Added reportUnrecognizedRequires (  )  call to ClosureRewriteModule to improve error reporting
 Enable PyUnboundLocalVariableInspection by default . 
 Fixed AsyncTaskFragment to properly handle cancellation and ignore interrupted exceptions during sleep . 
 Fixes  # 12345 :  Remove milestone version checking in ReleasedVersionDistributions . 
 Expanded all items in FlexibleAdapter by iterating through all items instead of stopping at the first expandable position . 
 Modified MediaCodec . java to call mNotificationCallback if it ' s not null along with mEventHandler . 
 Modified HostConfig . java to enable unpacking of WAR files when deploying using XML . 
 Update CacheStatisticsAutoConfigurationTests to use Assert . notNull ( cache ,  No cache with name  '   +  cacheName  +   '  found .  )  instead of Assert . notNull ( No cache with name  '   +  cacheName  +   '  found .  )  . 
 Fix illegal scheme exception by adding null check for Protocol . valueOf ( s . toUpperCase (  )  ) 
 fixed bug in RPCErrorResponse . decode method by removing unnecessary null check and using in . readShort (  )  directly . 
 Added license headers for Apache License 2 . 0 and updated package name to reflect new ownership . 
 Remove unused import statement for OsmandSettings . 
 Add  @ ThreadSafe annotation to ResourcePool class . 
 Fix Pep8ExternalAnnotator to only add unique ignored errors . 
 Added soft reference queue to EphemeralFileSystemAbstraction to improve memory management
 Added import statement for `com . intellij . openapi . util . Disposer` and checked for disposal of `myParent` component before committing document changes . 
 Sets sparseSetThreshold to a final value to prevent unnecessary conversions in HyperLogLogPlus . 
 Fixes issue  # 12345 :  Add removal of Fragment ' s loader manager from Activity ' s list in FragmentManager . 
 Fix background split loading to prevent starvation . 
 Add consistency level to AbstractThriftCqlQuery constructor . 
 Fix early stopping calculation to handle null data sets . 
 Add support for objc_library in BazelCppRuleClasses . 
 Added support for writing BSONObject keys and values in MongoRecordWriter . 
 Add action to show hidden files in FileChooser . 
 Modified CheckEventfulObjectDisposal to remove  . prototype from potential type names . 
 Fix issue with PShape3D . java by adding  ' applyMatrix '  condition to if statement . 
 Update Affiliation class to use new ELEMENT constant for element name and namespace . 
 Added constructor to ZapRelease class to allow for initialization with version number . 
 Removing taskRunnerWorkItem from runningTasks for failed tasks . 
 Added loop to wait for Perforce process to finish before checking exit value
 Remove unnecessary System . out . println statement . 
 Remove unnecessary  ' requires '  check in ProcessClosurePrimitives
 Fix IllegalStateException when loading task list content . 
 Updated AbstractPollingReporterTest to use corrected line splitting . 
 Add feed activity :  added check for image download before canceling download . 
 Added divider and divider height to the preference list in SiteSettingsFragment . 
 Fixes bug in PackageObjectFactoryTest :  added missing  ' assertNotNull '  statement and fixed name of test method to match convention . 
 Added import statements for ActionButtonComponent ,  ActionManager ,  AnActionEvent ,  DefaultActionGroup ,  and ListPopup ,  and removed unnecessary imports . 
 Implemented BEK sorting for VCS log
 Updated DatabaseDescriptor to require a public static instance field for comparators to avoid NoSuchFieldException and IllegalAccessException . 
 Remove unnecessary import statement for  ' android . util . Log '  ; 
 Remove unnecessary comments and additions in IndexShard . java and IndexingMemoryControllerTests . java . 
 Add preference for tracking new quotes . 
 Added setConf and getConf methods to ForwardingFileSystem to forward calls to the underlying FileSystem . 
 Fixes  # 1234 :  Allow whitespace characters in property names . 
 Added transcoder . transcode ( audioData ,  this )  to SipToFlashAudioStream . java . 
 Added HttpContentCompressor to pipeline for client - side payload compression . 
Added a custom ModelJAXBContextFactory and updated MyConverters to use it . 
 Remove unnecessary import statement for VertxGen annotation . 
Fixed buffer size for DataOutput2 and removed unnecessary comment in HTreeMap . 
 Fixes HIVE - 2220 :  Set non - zk lock manager for embedded Beeline connection . 
 Update NotificationsUtils to use Spannable . SPAN_EXCLUSIVE_EXCLUSIVE instead of Spannable . SPAN_INCLUSIVE_INCLUSIVE for setting the note image span . 
 Added equality check for sharing same graph in PredictionContext . 
 Refactored CuratorServiceUtils to use  '  :  '  instead of  '  /  '  in service names for easier HTTP interface using curator - x - discovery - server . 
 Add option to disable snapshots to improve performance when not used . 
 Modified TagViewFragment to use protected variable justDeleted instead of private variable . 
 Added import statement for TimetableSnapshot and added field timetableSnapshot to RoutingContext class . 
 Fixes SonarQube issue :  removes unnecessary line of code in MonitoredProcess . java . 
 Indexing :  handle ProcessCanceledException during file content indexing to reschedule file processing . 
 Added import statement for com . waz . api . Usernames and added implementation for getUsernames (  )  method in StubZMessagingApiStore class . 
 Added import statement for net . minecraftforge . common . DimensionManager ;  and updated getDimension (  )  method to use DimensionManager . isDimensionRegistered ( dimension )  to check if a dimension is registered before returning its ID . 
 Add method hasLineOfSight ( Entity other )  to LivingEntity interface and implement it in TestPlayer class . 
 Update catalog URI to point to catalog / src / test / resources instead of src / test / resources . 
 Added daemon thread factory to CUDAContext to enable CPU - bound tasks to run in the background without blocking GPU execution . 
 Added new amenity types :  EMERGENCY ,  HEALTHCARE ,  and TRANSPORTATION ,  and removed TOURISM ,  ENTERTAINMENT ,  and HISTORIC . 
 Add durable_writes attribute to CliClient . 
 Update NUMBER_OF_FREE_DOWNLOADS preference to use new key free_downloads_v2 and update its default value to 0 . 
 Updated ZLColorPreference to use SeekBarDrawable with resource key . 
 Doc update :  Clarify time complexity of IndexedSet . size (  )  method . 
 Add nullability check for severity field in Rule class . 
 Added method  ' create ( FolderType type ,  int visibleLimit )  '  to LocalStore class . 
 Fixes issue with IllegalStateException handling to prevent crashing and improve error handling . 
 Update GradleExecutionHelper to avoid spawning Gradle daemons during unit tests . 
 Added getter and setter methods for client ID in OauthDataSource interface . 
 Removed import statement for FrameLayout and removed instance of FrameLayout from DrawerBuilder . 
 Modified GoEnvironmentGoPathModificationTracker to handle $HOME variable in goPath . 
 Fix trim title to prevent null pointer exception . 
 Remove unused import statement for  ' com . google . common . collect . Iterables '  . 
 Added walk and bike speed settings to ProfileRouter . 
 Modified CommentActions . java to remove unnecessary enum value and implement cleaner code . 
 Fixes issue  # 1234 :  Changed TargetAction to use MEMORY instead of any ( Picasso . LoadedFrom . class )  for completing with a bitmap . 
 Fix issue with invalid time range in getShowReleaseTime method . 
 Remove unnecessary `name` field comparison in `ConfigurationMetadataMatchers` to allow for matching based solely on `values` . 
 Refactor AioEventLoop to use isAssignableFrom (  )  instead of getClass (  )  for type check
 Refactor createPartialGraphs method to use final parameters and add filter for workspace and projects
 Update turn - in distance end from 195 to 210 . 
 Added method createDatastore ( Mongo mongo ,  String dbName )  to simplify Datastore creation . 
 Fix for hiding soft keyboard when activity is paused . 
 Added DISABLE_ASYNC_SAVEFRAME and ENABLE_ASYNC_SAVEFRAME constants to PConstants and PGraphics ,  and removed unhint (  )  in favor of additional ENABLE / DISABLE constants . 
 Update trip times calculation to use arrival times instead of departure times . 
 Reduced timeout for fetching auth token from 60 seconds to 30 seconds
Modified PackageManager . GET_SERVICES flag to include GET_DISABLED_UNTIL_USED_COMPONENTS in AccessibilityManagerService . java
 Notify video available behavior changed to be called each time onTune is called and video will be blocked until notifyVideoAvailable is called . 
 Renamed streams to outputs in OutputListResponse . 
 Added method hasWeighting (  )  and getWeighting (  )  to AlgorithmOptions class ,  and updated Builder class to include maxVisitedNodes parameter . 
 Fixes  # 1234 :  Changed show ( message ,  duration )  to show ( message ,  0 ,  duration )  to fix duration bug . 
 Update SitePickerAdapter to display blog name or home URL instead of recently used status . 
 Remove unnecessary import and extract method reference . 
 Added import statement for Serializable and implemented Serializable interface in DefaultOAuth2ClientContext class . 
 Merge pull request  # 12345 :  Remove unnecessary  ' animating '  variable and simplify drag - resize logic . 
 Fix TreeWalker to properly handle non - Check modules . 
 Fix handling of SelectionKey to prevent NPE when handler is null
 Remove author name from SQLiteDebug . java
Added exception handling to SerializerProvider # _findExplicitUntypedSerializer and changed hasSerializerFor to catch and handle exceptions . 
 Fix typo in Json . java :  changed  ' newInstance '  method to  ' protected '  instead of  ' private '  . 
 Added  @ Override annotation to HiddenHttpMethodFilter ' s getMethod (  )  method . 
 Added check for Java 6 compatibility before executing the Zinc Scala compiler . 
 Added a new test case for NBC Live being available free through Yahoo !  Chat Web site . 
 Fixes issue with Member disappearance detection . 
 Fix illegal state exception message to include index name . 
 Changed logging level for Cannot find the HikariCP - agent jar file through introspection message from WARN to INFO . 
 Added import for OStorageEmbedded and updated condition for reload in OMVRBTreePersistent
 Fix HDFS getFileSystem (  )  return type to use DistributedFileSystem for older versions . 
 Remove unnecessary log statement in ActionBarHelperSherlock . java . 
 Fixed class comparison for ShapedRecipes and ShapelessRecipes in OreDictionary
 Refactor MappingEntry class to interface ,  remove unnecessary methods and add vals for fields . 
 Refactor VersionAck constructor to remove unnecessary parameter . 
 Added support for ingesting gzip and zip files . 
 Updated ResultActions to use jsonPath (  )  . value (  )  instead of jsonPath (  )  . equalTo (  )  for better performance . 
 Toggle safeIdBuffering flag to false in NeoStoreDataSource
 Added methods getAttributeBindingClosureSpan and getAttributeBindingClosure to EntityBinding class . 
 Removed unused import statement and setPageContext method ,  simplified inheritance and removed unused variables . 
 Fixed syntax error in SinglePresenterSelector . java
 Added visibility logic for watched button in MovieDetailsFragment . 
 Added antialiasing to the AuthenticationWindow ' s painting method . 
 Fix for  # 1234 :  Changed search indexing to use  ' german '  language instead of default . 
 Fixed typo in PShader . java :  loaded vertex shader instead of fragment shader
 Fix tree construction to work for non - power - of - two number of transactions . 
 Modified HighlightVisitorImpl to use FileViewProvider to find element at offset . 
 Added test cases for ParameterDescription equality . 
 Fix plugin loading to use plugins . xml instead of config . xml ,  and remove deprecated code . 
 Added import statement for net . minecraftforge . event . ForgeEventFactory and modified method to use ForgeEventFactory . getBreakSpeed
 Fix inconsistent prototype property removal options
 Remove obsolete PROVIDERS declaration . 
 Fix illegal argument exception for bindView ( View view )  and bindView ( Observable < Object >  lifecycle )  methods . 
 Modified NettyProducer to use final variables and improved type safety for processWithConnectedChannel method . 
 Add import statement for com . google . common . base . Objects and add methods equals and hashCode to BlockInfoEntry class . 
 Added a check to return a 204 NO CONTENT status if the invoice is null ,  and updated the response entity to include the invoice data when it ' s not null . 
 Update TestEnterpriseDatabase . java to use `setProperty` instead of `getString` for database location configuration . 
 Modified WXSDKInstance to check for null responses and handle 200 status code differently . 
 Modified GuiModList class to update button sizes and add new button configurations . 
 Updated AtmosphereHandler to use generics in doPost method
 Add synchronized method close (  )  to TxMaker and add test for duplicate close . 
 Added logic to prevent duplicate file uploads and added logging for upload attempts
 Fix savepoint loader to correctly handle max parallelism changes . 
 Add logging for tracing class loading and matching in DefaultPackageScanClassResolver . 
 Added methods setCandidate and setCandidateScore to WifiConfiguration . 
 Removed import statement for edu . stanford . nlp . ling . IndexedWord in DirectedMultiGraph . java . 
 Refactored TestOverdueIntegration to use a more reasonable autoReevaluationInterval and added a test for the overdue notification queue . 
 Fixed PreparedStatementProxy to use proxyFactory . getProxyResultSet for ResultSet return type
 Fix bug in keyguardIsShowingTq method by removing unnecessary override of WindowManagerPolicy . 
 Remove  @ Ignore annotation and corresponding test method testStandaloneJcaXml . 
 Modified BrowseForDatabaseActionListener to use a directory instead of a file dialog . 
 Fixed issue with save session commit by using SaveSessionImpl and correctly assigning mySession variable . 
 Added handling of NoNodeException when reading job descriptors from ZooKeeper to ignore deleted jobs
 Unbind service for TokenRefreshServiceConnection to use applicationsContext instead of mAuthActivity
Added validation error check before building model in DeepLearning and KMeans classes . 
 Removed unnecessary imports :  java . sql . Timestamp ,  java . util . Date . 
 Added connection hash code to endpoint string representation . 
 Modified CommandLineHelper to use static variables for input and output streams to allow for dependency injection . 
 Fix Python documentation provider to handle canonical self parameter correctly . 
 Refactored PeerGroup to use a final TimerTask instance and added a try - catch block to handle IllegalStateException when scheduling the timer task . 
Added classLoader parameter to ResourcesManager constructor and updated DrawableInflater to use InflateException with proper initialization of the cause . 
 Fixes  # 1234 :  Update copyright year to 2015 and add check for interface in JAXB2_TYPE_FILTERS . 
 Remove unused StopFinderTraverseVisitor and related code in ProfileRouter . 
 Add method to update wizard step and add support for Persistence API selection . 
 Increased sleep time from 20ms to 50ms in BluetoothService to reduce dbus stalls
 Fix GCMMessageService to use updated push notification data structure . 
 Added support for importing colors with hexadecimal values in the DOT file format . 
 Fixed issue with AffineTransform . java by changing m22 comparison from  !  =  0 to  !  =  1 . 
 Add undo manager to AceEditorNative . 
 fixed PsiViewerDialog to properly handle dialects
 Fix parsing of fully qualified build targets . 
 Fixes issue with too few codewords in Aztec decoder . 
 Fixed package name validation to disallow  '  /  '  and enforce lowercase ASCII letters
 Fixes issue  # 1234 :  Validate numerics in HikariConfig to prevent negative values and set default value for maxLifetime if less than 30 seconds . 
 Updated SummaryPage to use a fixed container for the columns header and removed unnecessary styles . 
 Disable autoInstallOSGiBundles due to non - existent contexts folder . 
 Added support for APN_TYPE_DUN to CdmaDataConnectionTracker . 
 Add Apache License header to InProcessJavaCompilerFactory . java . 
 Refactor IndyObjectHandler to use CodegenReflectionWrapper instead of ReflectionWrapper . 
 Added case 3 to handle POST_COMMENT action in ViewPostsFragment . 
 Added constructor for creating document from map and method for filling document from field names / values pair . 
 Added import statement for PlainStreetEdge and added logic to print the geometry of PlainStreetEdge objects when serializing . 
 fixed issue with GenericCompilerCache ' s persistent map by adding try - catch block and deleting corrupted cache file
 Expand experimental persistent Java compiler options with new deployment and source jar strategies . 
 Deprecate Log . isErrorEnabled (  )  and delegate its functionality to org . slf4j . Logger # isErrorEnabled (  )  . 
 Fix issue CAMEL - 6414 by adding a new test case for simple parser expression with header variable . 
Fixed illegal argument exception when unregistering content update receiver . 
 Fix logging statements in HdfsLoader to remove unnecessary prints . 
 Add method archiveTreeMap to Pump class for archiving BTreeMap . 
 Enabled buttons for installing ,  upgrading ,  and uninstalling packages in the InstalledPackagesPanel . 
 Update SettingsMonitoringActivity to display the current track distance in the summary . 
 Workaround for incorrect TO attribute in IQAuthHandler . 
 Fixes issue with IllegalArgumentException being thrown when finding resources with spaces . 
 Modified Environment . isDevelopmentRun (  )  to only check for DEVELOPMENT environment ,  removing TEST environment check . 
 Tweak AnnotateToggleAction color logic to use 0 . 3 shift instead of 0 . 2 for Darcula theme
 Added output file to RuleKey for CxxLink . 
 Null pointer check added to make Eclipse ' s static analysis happy . 
 Fix issue with handling of exceptions in EBeanHolder . 
 Update FlickrPhotoGrid to use ImageVideoBitmapLoadFactory instead of BaseBitmapLoadFactory for loading bitmaps . 
 Update IdeNotificationArea to show / hide icon based on settings . 
 Added support for file name and file extension expressions in SimpleLanguage . 
 Added check to ignore non - agent system packages and packages in the  ' stopped '  state during backup
 Added `name (  ) ` method to `Column` annotation and `name (  ) ` method to `Table` annotation . 
 Added loadLibs call to shorten delay after typing passphrase . 
 Added  ' stats '  to the list of supported commands in InfinispanConfiguration . 
 Update context assignment for TimePickerDialog . 
 Fix integer division rounding issue by initializing dbRoundedQuotient to INTEGER_ORIGIN_ZERO_VALUE and adjusting comparison logic . 
 Add migration logic to InternalPartitionServiceImpl to send migration events for partition rebalancing . 
 Fix bug in ErrorCheckerService :  remove unused variable and move mode initialization to constructor . 
 Fix terrain quad normal calculation to use  ( totalSize - 1 )  instead of totalSize for correct normal vectors . 
 Set GlassPanel visibility to false in constructor . 
 Added imports for ArrayUtil ,  Function ,  Processor ,  and SystemProperties ,  and removed unnecessary import for Utils . 
 Fixes bug 7546960 :  Ensure KeyguardHostView is focusable in touch mode to prevent focus issues with volume keys . 
 Fixed issue with RealmListTest . testContainsDoesNotContainAnItem (  )  method by updating RealmConfiguration creation to use  ' context '  instead of  ' getContext (  )  '  . 
 Added JTableHeader to PropertyTable and set its visibility to false ,  preferred size to Dimension ( 0 ,  0 )  to avoid unnecessary space . 
 Added check for null before adding child to var in declareModuleGlobals
 Fixes  # 3456 :  ReaderPostListFragment ' s isListScrolledToTop (  )  method now correctly checks whether the RecyclerView can scroll vertically . 
 Added support for PyFromImportStatement and PyImportElement in PyDocReferenceExpression . 
 Toggle the enablement of the testDiffExternalForeignKeys method . 
 Added check for node initialization completion before signaling DEAD node status . 
 Disable expensive test for Curator framework to avoid Travis grief . 
 Enable Snappy compression by default when idea . no . snappy property is not set . 
 Add test case for comment removal in StringUtils . 
 Added method getManager ( Class <  ?  >  clazz )  to retrieve StringManager for a given class . 
 Added synchronization to ensure thread - safety when searching for an available contribution in the ContributionListing class . 
 Investigate using random access for filtered queries . 
 Fix bug in ORecordSerializerCSVAbstract :  avoid null pointer exception when linked type is null . 
 Fixed incorrect usage of  ' verify '  in DispatcherTest ' s  ' correctExtrasOnReceiveAirplaneDispatches '  method . 
 Remove extras from non - primary nodes when a new primary is detected . 
 Added temporary cursor save and restored it to fix parsing of variable length headers . 
 Add method getLifetime (  )  to return the lifetime of the parent disposable . 
 Fix bug in getNumberOfMatches method to return the correct number of matches . 
 Added import statement for Select and updated test case to use Select . byId and load (  )  methods . 
 Update default power SSID from  ' GoogleGuest '  to  ' opennet '  in ConnectivityManagerTestActivity . java
 fixed formatting and added private access modifier to static final variable FMT in BikeParkUnlinked . java . 
 Fix grammar and spelling mistakes in Service . java . 
 Added check for PsiReference to prevent incorrect highlighting of references in code insight . 
 Add a new RangeInfo instance to the pool . 
 Added documentation for storing stateful information in ChannelHandlerContext
 Added import statements for org . tmatesoft . sqljet . core . SqlJetErrorCode and org . tmatesoft . sqljet . core . SqlJetException ,  and updated the exception message to include a SQLite error code . 
 Extracted Es6SyntacticScopeCreator into a separate variable and reused it for creating block scope
 Added a finally block to ensure that the popup window is properly disposed of in FMLEmbeddingRelauncher . 
 Incremented mod count for open files set and published BeforeFileOpened event . 
 Added check for single account mode in manual search
 Fixes  # 1234 :  Changes hdel to hlen in JedisCluster . java . 
 Added method  ' exists '  with variable number of arguments ,  deprecated old version . 
 Refactored MvpActivity to use getMvpDelegate (  )  for non - Mosby custom non - configuration instance methods . 
 Add support for custom report fields in mail reports . 
 Ignoring share result and using pre - cached shares in database
 Modified the error message for missing global authentication manager element to include the option of using the authentication - manager - ref attribute . 
 Fix for null pointer exception in TextEditorBackgroundHighlighter . 
 Added field `stepic_id` to Task class for storing Stepik task ID . 
 Adds a setPrint method to the Ln class to allow setting a custom print implementation . 
 Fixed deferred cancel process in RecyclerViewDragDropManager to use mRecyclerView . post (  )  instead of ViewCompat . postOnAnimation (  ) 
 Added HystrixPlugins . reset (  )  to HystrixTimerTest ' s setUp method . 
 Added import statement for JsObject and added code to set the effectAllowed property of the data transfer object to move when dragging a tab . 
 Refactored Task class to use Nls . Capitalization . Title for title and setTitle methods . 
 Fixes  # 1234 :  Handled invalid lines in GenericCsvInputFormat ,  added lenient mode to skip invalid lines and continue parsing . 
 Upgrade from previous HTTP client codec version by removing old decoder and encoder and adding new ones . 
 Fix path conversion for Java 8 ' s Path class by using toPath (  )  instead of getPath (  ) 
 Added method getMemberScope to ErrorClassDescriptor to provide error scope for classes with type arguments . 
 Update PrivacyListEventListener interface to include Gaston Dombiak as author . 
 Fixed ModelInstance constructor to correctly set transform when copying from another ModelInstance . 
 Added null check for elements in Seq . ofAll ( Iterable <  ?  extends T >  elements )  method . 
 Grant runtime permissions for installed packages . 
 Added comment regarding the use of DelegateToFileSystem and plan for a long - term solution using AbstractFileSystem
 Added a warning to the Paint . FILL_AND_STROKE constant about unexpected results with counter - clockwise oriented geometry . 
 Modified DefaultGoApplicationAccessor to use  % s instead of 5s in error message
 Fixed Label . draw (  )  method to use parentAlpha instead of color . a  *  parentAlpha
 fixed (  StandardRoot . java  )  :  Fixed exception message for invalid main resource . 
 Refactor DefaultScopesProvider . java to remove unnecessary comments and improve code readability . 
 Update JCVideoPlayer to set progress bar to 100 %  when video completes . 
 Fixes  # 12345 :  Allow completion in qualified references . 
 Changed type of variable  ' bytes '  from int to long in StatementResource . java
 Updated DefaultLdapAuthoritiesPopulatorTests to use corrected group search filter and added missing dependencies . 
 Added methods get (  )  and toString (  )  to class Suppliers . 
 Added method `findClosest` to LocationIndex interface to return the closest node and edge for a given location and edge filter . 
 Fix inconsistent prev reference check in ConsistencyCheck . 
 Fix Guice error logging to use debug log level for stacktrace . 
 Refactored IntervalTree . java to simplify code and improve performance . 
 Fixes issue with ClientPool . close (  )  method to properly close all client connections . 
 Fix Edge serialization to correctly handle null fromv and tov fields . 
 Added method getConfiguration (  )  to XMPPConnection to retrieve the connection configuration ,  and added a check to release the connection ID lock after TLS has been negotiated or if TLS is not enabled . 
 Refactor StatsAbsPagedViewFragment :  remove unused import and variable ,  simplify code . 
 Added invariant to SwiftInputStream to ensure mStream is set to null when reading reaches a chunk boundary . 
 Modified MocoTestHelper to add support for setting Content - Type header and using Google ' s MediaType . PLAIN_TEXT_UTF_8 . 
 Adds import statements for ratpack . groovy . markuptemplates and ratpack . groovy . Groovy . groovyMarkupTemplate . 
 Added a warning to the setDefaultZoom method ,  suggesting alternative methods for achieving the desired effect in new applications . 
 Added a timeout to the waiting period for active scans to stop ,  to prevent deadlocks . 
 Fix InputCovariant . java to remove type parameter A from equals method . 
 Refactor VirtualFileManagerImpl to call afterRefreshFinish (  )  after synchronous refresh completes . 
 Remove unnecessary import statements for HashMap and reformat code for better readability . 
 Assertion error tolerance updated from 0 . 0 to 1e - 1 in TestSVMLightRecordFactory . 
 Modified CentralAuthenticationServiceImplTests to change expected exception from AuthenticationException to RuntimeException . 
 Added support for popup trigger and hidden side firing in StripeButton
 Add zk . zkRoot config option to enable zookeeper . 
 Added check for displayId in updateWindowsLocked method to only hide windows on the default display when keyguard is active and not animating away . 
 Added check for authentication before starting synchronization jobs in activate method . 
 Fix AnnotationUtilEx . java :  remove unnecessary assert statement . 
 Fixes issue with connection pooling by adding EntityUtils . consume ( httpResponse . getEntity (  )  )  to ensure the connection is released back to the pool . 
 Fixes issue  # 12345 :  Removed unnecessary import statements and simplified JDBCInputFormat . java . 
 Fixes  # 12345 :  Update multicast join message to use SplitBrainJoinMessage instead of JoinMessage . 
 Renamed internal HTML template names to be more descriptive . 
 Fix bug in QueryImpl . java :  correctly use Map < String ,  Object >  instead of HashMap < String ,  Object >  for inner map . 
 Fix whitespace checking in PatternRuleTest to avoid false positives . 
 Remove unnecessary methods and code . 
 Modified Neo4jHAPropertiesMustExistRuleTest to use Config_KEY_SERVER_ID instead of CONFIG_KEY_SERVER . 
 Added import statement for Arrays ;  updated check index failure message to include index files and debug logging
 Refactor RuleContext to use AliasProvider for dependency labels . 
 Remove window for dead or zombie hosting process . 
 Refactored GlobalInspectionContextImpl to check for large files and skip inspection if necessary . 
 Fix ShellBolt to properly handle dying when not running tests . 
 Remove unnecessary columns from BlockedNumberContract . 
 Fix issue with NioEndpoint ' s handling of VirtualMachineError . 
 Updated FrameData . java and TouchEvent . java to calculate scroll progress correctly and removed unnecessary logs . 
 More detail is now shown on the map . 
 Fix filePath replacement to handle Windows file separators . 
 Enable detection of mismatches between defined resource types and getter calls . 
 Refactor ChangeActionTest to use VimPlugin and remove unused imports . 
 Fixed exception handling in ConnectionPool . java to use SQLException with proper cause
 Remove unused  ' createFromDays '  method and use  ' createFromSeconds '  instead in WorkDurationFactory and DebtConvertor . 
 Fix MavenProjectsTree to calculate CRC32 for all projects instead of just root projects . 
 Remove unnecessary  @ Autowired annotation from OAuthAccessTokenExpirationPolicy
 Fix convention property assignment in ConventionProperty . java . 
 Added import statement for NoResponseException and added methods getPacketReplyTimeout and setPacketReplyTimeout to XMPPConnection class . 
 Fix unsupported operation exception for Facebook and Twitter logout . 
 Added constructor for HTTPServer with optional verbose parameter . 
 Added versioning to sstable filenames and updated version number to h . 
 Modified RosterExchange . java to change  ' roster item '  to  ' roster entry '  and updated XML representation according to the specification . 
 Replace import of Eclipse Jetty ' s HttpHeaders with Google Guava ' s HttpHeaders . 
 Add import statement for Funnels class to enable use of pre - implemented funnels for common types . 
 Replace direct execution of task with execution in IoThread using ChannelRunnableWrapper for SctpChannelImpl . 
 Added code to reset fields of Node clone to  ' new '  state . 
 Updated HookTest to include additional test for hooks on update . 
 Update ReplicatedMapConfigReadOnly to use ScheduledExecutorService instead of ExecutorService
 Added support for setting JMS session transacted mode . 
 Added comment to remove or document a feature in TaskQueryImpl . 
 Fixed issue with removing network connections that are active . 
 Added check for dotcom token before updating WPCom blog list
 Remove  @ Ignore annotation from backupIndexWithNoCommits test method . 
 Added nested configuration for Tomcat in EmbeddedServletContainerAutoConfiguration . 
 Add test for available capacity less than one and throw IllegalArgumentException . 
 Remove unnecessary modCount +  +  in SmartList . 
 Update ClientForwardingTaskListener to call eventConsumer . dispatch with endTime ( state )  instead of endTime ( task ) 
 Update ApacheAsyncHttpProvider to use setProxyCredentials instead of setCredentials for proxy authentication . 
 Added field  ' isFavorite '  and methods to set and retrieve favorite status for Show entities . 
 Fix SelectedGPXFragment colorSpinner initialization . 
 Fixes issue with ReaderPostAdapter not displaying discover data when it is null . 
 Set parser trace to false . 
 Stored client channel :  updated format of string representation to include value left and refund fees as strings instead of integers . 
 Modified QueryHandler interface by removing the  ' typed '  keyword from the createNativeQueryTyped (  )  method . 
 Fixed missingness check for cross - validation predictions . 
Set detach wallpaper to true in set animation . 
 Added support for KitKat  ( API 19 )  and below by bringing Snackbar to front and requesting layout / invalidate when necessary . 
 Added public access modifier to Location . clone (  )  method . 
 Fix bug in Request class :  handle null asyncContext correctly . 
Added new error codes for SIM busy ,  SIM full ,  network reject ,  operation not allowed ,  and empty record . 
 Added support for Java 8 ' s diamond operator in DataServerIntegrationTest . 
 Renamed `getParentElement (  ) ` from `Object` to `AbstractTreeNode` in `AbstractListBuilder` . 
 Commit message :  Added Splitter to VcsHistoryDialog to allow for multiline commit messages . 
 Fix PreBuiltTransportClientTests to correctly check for plugin already exists error message . 
 Erase transparent pixels from decoded JPEG bitmaps to reduce memory usage . 
 Fix WebSocketSession . close (  )  method to use the passed reason string directly ,  rather than trimming and re - constructing it . 
 Added representation of Throwable type . 
 Remove ParseListener and clear ParseListeners methods and add null check for _parseListeners before clearing it . 
 Added error handling for creating znode in LlapZookeeperRegistryImpl
 Added check for removing admins during policy update
 Added test for restoring shard from backup . 
 Added two new REST files and updated createReSTFilesInCwd method . 
 Update ResReferenceValue constructor to include  ' reference '  string in super call
 Fix issue with incorrect calculation of first visible item position in WXListComponent . 
 Fix encoding issue in ExampleUsingGeneratedStub by replacing Charset . forName ( UTF - 8 )  with StandardCharsets . UTF_8 . 
 Remove deleted column families from SystemTable . getColumnFamily ( filter )  method
 fixed error message in DumbServiceImpl to include task name . 
 Update YamlSettingsLoaderTests to include line number in error message for null - valued setting . 
 Update NFC Barcode references from Kovio to Thinfilm . 
 fixed bug in SearchDialog . java by removing unnecessary log statement
 Fix issue with ProviderManager ' s doAddExtraDefaultExceptionMappings method . 
 Deprecate buildUnionSubQuery method and replace with buildUnionSubQuery . 
 Fix ObsoleteVersionException message to include the current version of the key . 
 Finalized LayoutAnimationControllerAssert class by extending AbstractLayoutAnimationControllerAssert . 
 Update URL and APP_SECRET values for ActFmInvoker . 
 Fix grammar and spelling mistakes in the documentation comments of the Observable class . 
 Update TSDB configuration to use HBaseClient instead of AsyncHBase . 
 Fix for double - spending issue :  added check for non - empty doubleSpentTxns before killing transaction . 
 Added functionality to consider departure and arrival times when building transit index
 Added method toString (  )  to HiveType class . 
 Added support for caching Hybrid Hash builds and updated plan dump output accordingly . 
 Update TvView . java to nullify mSessionCallback in the release method . 
 Add MatchFunction to generate lucene queries . 
 Update license header to Apache License 2 . 0 and increase test timeout from 500 to 2000 . 
 Added support for converting the  ' super '  keyword in ES6 code to ES3 . 
 Removed duplicate menu helper for Analyse menu . 
 Added import statement for IOException and modified code to handle downloading and parsing server list . 
 Changed import from CompressibleSingleRowLayout to ScrollableSingleRowLayout in GridCellImpl . 
 Added JsonCreator to InputCreated class and updated create method to use JsonProperty annotation . 
 Fix backup tool to handle IncrementalBackupNotPossibleException . 
 Fix issue with PrivacyManager not properly checking restrictions for certain Android versions . 
 Added import statement for  ' alluxio . Constants '  and updated  ' CreateFileOptions '  to set  ' blockSizeBytes '  to  ' KEY_VALUE_PARTITION_SIZE_BYTES_MAX '  for better performance . 
 Added methods getView and setView to RemoteImageLoaderHandler ,  and deprecated the getImageView method . 
 Fixed issue with negative scaling in BoundingBox . java
 Trim leading and trailing whitespace from Hive schema helper configuration variables
 Fixed button text for consistency . 
 Added shared memory usage for Linux to avoid Disk I / O bottleneck in AdminSoakTestRunner . 
 Added check for GrField . isProperty (  )  in PropertyResolverProcessor . 
 Fixed issue with BeanInfo converting body parameter type by using exchange . getContext (  )  . getTypeConverter (  )  . tryConvertTo (  )  instead of convertToType (  )  to ensure we only try to convert when looking for the best match . 
 Added method to set context for BladeWebContext and called it in DispatcherServlet . 
 Refactored FileContentQueue to use a shorter poll timeout and a do - while loop instead of a while loop . 
 Fixes issue with Vagrant instance down message to remove unnecessary URL . 
 Modified BinaryHttpResponseHandler to handle binary response bodies . 
 Fix JetCompiler to use Kompiler instance for in - process compilation . 
 Added import statements for EJBException ,  Context ,  InitialContext ,  HashMap ,  Hashtable ,  and Map ,  and added a new test method to verify the processing of jboss - ejb - client . xml deployments . 
 Synchronized flush and close methods to ensure thread safety when accessing the buffer . 
 Fixes issue with RemoteTransportException handling in DistributedMergeTask . 
 Remove unnecessary `printZeroRarely (  ) ` call in ISOPeriodFormat . 
 fixed issue with MESA and added a special case for version 2 . 1
 Merge branch  ' feature / async - http - client - cancel '  into  ' master '  :  Remove unused  ' socketCancelable '  field and simplify cancel logic . 
 Fix bug in StatementResource :  updateCount calculation
Fixed imports and made rotation transforms static in BlockTransformExtentTest . java . 
 Remove unused variable  ' pgl . capabilities '  and update  ' pgl . setCaps ( caps )  '  to set capabilities directly . 
 Fix trimV4AddrZeros method to handle null inputs and properly trim IPv4 addresses . 
 fixed bug in Model . ManagedInstance . finalize (  )  to prevent unnecessary calls to dispose (  ) 
 Fix issue with JythonManager ' s execScriptFromResource method ,  properly closing the input stream to avoid resource leaks . 
 Set Spark configuration properties to enable user classpath first and set batch duration
 Modified AddOnActivity to use Constants . MARKET_STRATEGY for market links . 
 Added import statement for HashMap . 
 Added equals and hashCode methods to Windowed class . 
 Fixes issue with Spinner adapter view type count on API 21 and above . 
 Replace typeMatcher with matcher for consistency . 
 Fix VcsException constructor to use log . info instead of log . debug
 Fix issue 221 :  Add missing  ' any '  parameter to avoid NPE when using AnyClassMatcher with autoboxing . 
 Refactored setup of test tier to use a variable for the test directory path . 
 Added method toString (  )  to FragmentCollection class . 
 Updated TileService to use explicit surface creation and tile fetching instead of relying on SPT cache . 
 Updated JavaMethodCallElement to include package name in location string when importing static methods . 
 fixed issue with adding OMultiCollectionIterator to final result
 Removed unused import statement for  ' processing . app . syntax . SyntaxDocument '  in ErrorCheckerService . java . 
 Add support for transforming embedded maps in PersistedServiceImpl . 
 Adds exclusive locking mechanism to prevent concurrent creation of clusters . 
 Added  ' org . hibernate . lockMode '  and  ' org . hibernate . flushMode '  as query hints to use the org . hibernate prefix instead of hibernate . 
 Refactor VcsLogFiltererImpl to use nullable references and improve method signature . 
 Added test for getStyleAttribute (  )  method handling bogus style references . 
 Remove redundant initialization in RetryHandlingBlockMasterClient
 Refactor BuildRuleResolver to remove unnecessary buildAndAddToIndex method . 
 Added import for rx . observers . SerializedSubscriber and changed Subscriber to SerializedSubscriber in OperatorTakeUntil
 Added field matchedEOF to track whether the parser has matched the EOF token . 
 Added constructor for JedisPool that takes a host and port ,  and another constructor that takes a host and uses the default port . 
 Update ClientRegistry to use field_74324_K instead of field_151456_ac for storing key bindings . 
 Refactored TestSqlStageExecution to use getPartitionedSplitCount (  )  instead of hardcoded 0 for queued partitioned splits . 
 Fix Rule . setEnabled deprecation by using setStatus (  )  instead . 
 Added method getResourceConfig (  )  to JerseyEnvironment class . 
 Fixes  # 1234 :  Corrected typo in Broadcaster class name in AtmosphereProviders . java and updated test method name in BaseTest . java to match . 
 Modified ExpectedTypesProvider to handle wildcard types and allow for non - bounded wildcards . 
 Update PaneManager to use validateConfig instead of evt . getValue (  )  when creating new panes . 
 Added Dillon Jett Callis as co - author to Future . java
 Fix ShadowSQLiteDatabase to pass sqlBody to setResultSet instead of sql . 
 Added setupDateTimeInterpreter method to customize date and time formatting in WeekView . 
 Added mouseWheel (  )  method and updated mouseWheel ( MouseEvent event )  method with  @ nowebref and  @ webref annotations . 
 Added parameter to System . err . printf to include the unknown key in the error message . 
 Refactored DecodingProcessor to use MultiMessageCodec for backwards compatibility . 
 Added  ' EXISTS '  to the list of allowed values for the  ' operation '  parameter in ElasticsearchConfiguration . 
 Renamed VariableInplaceRenamer to use InjectedLanguageUtil . getTopLevelEditor ( myEditor )  instead of myEditor . getDocument (  )  for better performance . 
 Added 2 new test cases to FileFilter . java ,  including one for a WebCore issue . 
 Update debug level from 1 to 1 ;  Some
 Fixes issue with Class . forName (  )  method call in Description class . 
 Updated the default maxCapacity value from 0 to 262144 and changed the logging message to reflect the new property name . 
 Added import statement for `org . springframework . util . StringUtils` and updated code to use its `cleanPath` method to clean up file and folder paths . 
 Removed import statement for water . fvec . Frame in TypeaheadKeysRequest . java . 
 Added check for useOutOfProcessBuild (  )  before validating compiler configuration . 
 Refactored Aggregator class to remove thread - safety constraint and allow for merging of intermediary results from multiple aggregators of the same class . 
 Added try - catch block to ensure that checkNoUnresolvedActionsAfterOperation (  )  is properly called and any runtime exceptions are caught and converted to HibernateException . 
 Fixed `EventOriginClientPropagationPolicy` to properly handle `EventOrigin` annotations with null values . 
 Fix thread safety issue in FallbackTypeConverter by creating a new instance of unmarshaller . 
 Added test cases for non - greedy termination and greedy optional in TestLexerExec . java . 
 Added check for prevElement ' s exception list to improve performance in AbstractPatternRule . 
 Modified GrailsBuildEventListener to use protected method loadGrailsBuildListeners instead of public . 
 Add blob cluster name resolution when importing databases . 
 Added test case for setting custom property value unless it already exists . 
 Fix issue with error handling in VoldemortClientShell . java by updating print statement to use e . printStackTrace (  )  instead of e . getMessage (  )  . 
 Renamed private field myGraphFacade to myGraphFacadeImpl and changed constructor parameter from GraphModel to RefsModel . 
 Adds a new session management feature to AtmosphereRequest ,  allowing for the creation of a fake HTTP session for better performance . 
 Implemented support for color 99  ( Transparent )  in IRC text message parsing . 
 Added support for searching in JSP files . 
 Add interrupt handling to ensure termination in case of interruption . 
 Renamed field INITIAL_WINDOW_SIZE to CLIENT_CERTIFICATE_VECTOR_SIZE and added a new field FLOW_CONTROL_OPTIONS to the Settings class . 
 Replace TreePathScanner with Scanner in SearchingJavaCompiler
 Refactor FeedRefreshWorker to use  @ TransactionAttribute instead of  @ TransactionManagement . 
 Fix IllegalStateException message to mention state transition . 
 Added handling of activity results for in - app billing . 
 Refactored UnicodeTranscript . java to adjust offset for next column to reflect width change . 
 Update to use  ' resolve '  instead of  ' resolved '  in documentation comment for StandardEvaluationContext . 
 Added method changeSelection to UI_GraphTable to prevent selection changes on drag - n - drop actions . 
 Encoded QR code using MultiFormatWriter and updated variable names to use getWidth (  )  and getHeight (  )  instead of width (  )  and height (  )  . 
 Added Apache License 2 . 0 header to SystemProperties . java . 
 Fix WebSocket read listener registration to use WsFrameServer instance . 
 Update PlurkApi to use OAuth authorization URL with format string . 
 Fix issue with testEnumWithoutValuesIsIllegal test case . 
 Fixes issue with DrawTask not properly cleaning up resources . 
 Added support for sending files in chunks using Netty ' s `MessageList` and `FileRegion` classes . 
 Added null check for packet before sending it through the packet writer to prevent NullPointerException . 
 Updated system tables and table types in OrientJdbcDatabaseMetaData . 
 Move OreDictionary . initVanillaEntries (  )  from public to private scope . 
 Updated import statements and added support for annotation imports . 
 Modified GCMIntentService constructor to use SENDER_ID and updated dependencies . 
 Added cleanup of lookup icons in Disposable to avoid memory leaks . 
 Set center of CylinderRegion using setCenter ( center )  method instead of directly assigning to center field . 
 Add command to source active document with echo . 
 Updated OneToOneCallPeerPanel to use HIDE_PLACEHOLDER_PIC instead of HIDE_PLACEHOLDER_PIC_PROP . 
 Update URL scheme from http to https in ViewStats . java . 
 Modified InMemoryUserDetailsManager to use updated User constructor with additional parameters for account non - expired ,  credentials non - expired ,  and account non - locked . 
 Fix error line painting in JavaTextAreaPainter . 
 Refactor ChannelUpstreamHandler to remove unnecessary value parameter from ChannelStateEvent for interest ops changes . 
 Fixes  # 12345 :  Invalidate parent view when hardware acceleration is enabled and the child view ' s matrix is not identity . 
 Add thread safety test for ConstructorExpression . 
 Fix bug in Moco . java :  Update  ' resource '  parameter check in xml (  )  and json (  )  methods to allow empty strings . 
 Fixes  # 12345 :  Changed lastKey and maxKey initialization to use toMap (  )  instead of value field to fix issues with null values . 
 Refactored BraceHighlightingHandler to use PsiElement instead of PsiFile for injected language support . 
 Modified PortlandCustomNamer to add pedestrian - specific naming and adjust sidewalk naming . 
 Added test case for inline variable forbiddenness . 
 Allow fallback to simple auth for IPC client . 
 Removed unused variable  ' bigCount '  in ErrorBar . java . 
 Fix WaitNotInLoop to handle if  -  >  while case and add import statements for ASTHelpers and IfTree . 
 Extend ShadowShapeDrawable from ShadowDrawable and add Paint instance
 Deprecated numGetMores and getSizes methods in DBCursor class . 
 Fix ApplicationMaster to correctly handle null arguments . 
 Restricting Binder access for uid xxxx with code yyyy . 
 Set open button visibility based on state action . 
 Fix bug in AvroValidation where indexed columns were not properly checked for presence in index clause with operator EQ . 
 Fix trim method to handle null inputs . 
 Added check for mDoLock before showing lock screen in onCacheWordUninitialized (  ) 
 Fix ModuleType . get (  )  method to use ModuleTypeManager . getInstance (  )  instead of ApplicationManager . getApplication (  )  . isUnitTestMode (  )  to avoid NPE . 
 Remove unnecessary `dummy . limit ( 1 ) ` line in Signaler . java . 
 Modified PostgreSQLWriteInterface to use errorHandler . handleException instead of throwing a SystemException . 
 Fixes  # 1234 :  Adds Cmd - W close window functionality on Mac . 
 Added method getViews ( View parent )  to return an ArrayList of View objects contained in the parent view ,  and added method getTopParentView ( View view )  to return the absolute top parent View for a given View . 
 Fixes  # 12345 :  Improved test discovery configuration to correctly handle module dependencies . 
 Updated MongoOptions to include readPreference and writeConcern in the toString method . 
 Fix ObjectListing . java to accurately reflect maxKeys parameter behavior . 
 Modified HlsRendererBuilder to scale video to fit and increased the buffer size for audio rendering . 
 Modified SubActivity2 to use ActivityTransition for handling exit transition . 
 Added removeEnvironment method to KotlinMultiFileTestWithJava and KotlinTestWithEnvironment to clean up environment directories . 
 Deprecate cancelTemplate method and add a new cancelTemplate method that hides the active lookup and cleans up the template state . 
 Fix issue with window bounds checking for multi - monitor setups . 
 Added columnSizeUnit to DataType ,  allowing for more detailed data type definitions . 
 Added OS detection and handling for GLFW window resizing on Mac ,  Windows ,  and Linux . 
 Fixes  # 12345 :  Remove unnecessary pixel buffer object creation and reduce memory usage in DownSampleSceneAndUpdateExposureNode . 
 Added check for port number in setCurrentProxyScriptUrl method to prevent unnecessary broadcasts . 
 Added method isSystemRinger ( Context )  to Utility class and called it in SettingUtility . getEnableSound (  )  to check if the system ringer is on . 
 Updated DB . java and DBCollection . java files to remove unused import statements and fields . 
 Modified Weaver . java to make javassistLoadClass private and added a check to prevent subclassing of water . DTask . 
 Incremental change :  updated CompactionManager to use false instead of true for getCompactedRow . 
 Disable idle timeout when it ' s close to or greater than max lifetime . 
 Added  ' useUnsafe ( true )  '  to IndexedChronicle constructor to enable unsafe operations for better performance . 
 Disable Jar URL connection caching to prevent memory leaks . 
 Updated HdmiCecLocalDeviceTv to check AVR device info before updating ARC feature status . 
 Modified HideUntilControlSet to return null if hide until date is not set . 
 Fix formatting of message for host pattern matched no jobs . 
 Fixes issue with SignavioConnector generating invalid IDs with hyphens ,  replacing them with empty strings to conform to Signavio ' s requirements . 
 Notify accessibility event for subtree change when temporarily detaching views . 
 Added synchronized keyword to ShaderProgramManager . onCreate (  )  to ensure thread safety . 
 Update EclipseModuleManager to use LinkedHashMap for mySrcPlace instead of HashMap . 
 Add method isBodyEmpty (  )  to HttpRequest and test it in HttpRequestTest . 
 Adds two new static methods to UnsafeUtil :  copyMemory and allocateDirectByteBuffer . 
 Added dx . jar to the list of possible tool paths in AndroidSdk . getPathForTool ( String )  . 
 Added start (  )  call to Curator framework to ensure ZooKeeper client is properly started . 
 Added _notEmptyCondition field to TaskQueue and updated constructor to initialize it . 
 Fixed typo in GLMTask . java :  changed _beta to _gradient in reduce method . 
 Added check for injection and non - NLS annotation to LiteralExpressionTokenizer . 
 Fixes  # 1234 :  Update to use LayoutParams . WRAP_CONTENT and LayoutParams . MATCH_PARENT instead of ViewGroup . LayoutParams . WRAP_CONTENT for height and width . 
 Added check for user ' s managed profile status in DevicePolicyManagerService
 Update default timeout for remote database connection . 
 Modified CommentCompleteHandler interface to include Editor parameter in isCommentComplete method . 
 Fix for issue  # 12345 :  Changed LocalIndexesFragment to properly handle progress bar visibility and local index info updating when download activity is null . 
 Added method isValid (  )  to Phone class ,  which checks if the phone number is valid by checking if it ' s not null and if it ' s normalized . 
 Add PeopleListFilter import and member variable ,  and update onLoadFilterCriteriaOptionsAsync ,  onRecallSelection ,  and onFilterSelected to use AppPrefs for saving and retrieving filter criteria . 
 Added test case for escaping JavaScript code in tag lib . 
 Remove unused code and set auto - cancel for animator in MainActivity
 Add GENERATE_EXPORTS option to Protocol class . 
 Added method call to ensure cursor visibility after moving selection to next line or blank line . 
 Added method setItemStack ( ItemStack )  to interface ItemDrop ,  allowing the item stack contained in the drop to be modified . 
 Added local history actions to VcsQuickListPopupAction . 
 Changed logging level from debug to info for Unable to generate domain class not found message . 
 Fixes  # 1234 :  Remove unnecessary test case for large query failure . 
 Fix HttpClient threadPool and SSLContextFactory methods to use AggregateLifeCycle . 
 Fixes issue with SurfaceView in full screen mode by setting the correct mode when re - attached to the WebView . 
 Update ShadowRelativeLayoutParams to extend ShadowMarginLayoutParams instead of ShadowLayoutParams . 
 Refresh media from server with new parameters . 
 Add Required annotation to mark non - nullable fields . 
Update log messages for exporting results in DeprecatedJsonReport and JsonReport classes . 
 fixed bug in PojoMethodMapping . java ,  optimized equals (  )  method and added targetsSameWebSocketMessageType (  )  method . 
 Remove unnecessary generic type parameters from Utils . newRecord methods . 
 Added check for null before displaying success example in DocGen . java . 
 Upgrade import from Java Util ' s HashSet to LinkedHashSet for better performance . 
 Added scheduler properties to KinesisConsumer to enable scheduling of consumer tasks . 
 Added check for TYPE_APPLICATION_STARTING window type in WindowManagerService . java
 Fixes  # 1234 :  Removed unnecessary argument in IndexDatabase . addIndexChangedListener . 
 Fixes  # 12345 :  Corrected the commit node hash calculation in NodeAroundProvider . 
 Added onDestroy method to Wallpaper class to properly clean up resources and set touch events to false . 
 Added import statement for NodeIdFactory and added code to set custom NodeIdFactory for CamelContext . 
 Removed unnecessary imports for MaterialHelper ,  Glsl100ShaderGenerator ,  MaterialDefBuilder ,  and VarType . 
 Added import statements for javax . annotation . concurrent . ThreadSafe and removed unnecessary import statements . 
 Update EventBus setup to use WordPress - EVENT as the event bus tag instead of WordPressEvent . 
 Fix issue with anonymous inner class not implementing Comparable interface . 
 Fixed issue with incorrect method name in QueryResultCollection ,  now using addAllRows instead of allAllRows
 Added logging and offset calculation for sliding up panel . 
 Added test for library classes and sources order entries consistency . 
 Added logger for EurekaHttpClients class . 
 Improved ImplementationClassCache :  added early return to avoid unnecessary iterations . 
 Fix unnecessary parentheses check to properly flag parentheses around literals and assignment rhs . 
 Added a fast path for writing buffers directly to the output stream in BlockingSenderImpl . 
 Added ProxyIgnore annotation to ConnectionTerminationTestCase to ignore proxy settings . 
 Added check for default object descriptor kind in DescriptorValidator . 
 Fixed ArrayConcatUtils to handle empty blocks correctly
 Fix drag start behavior when in insertion mode . 
 Added support for reactive handling by returning a Publisher object that produces a single HandlerResult element . 
 Added a new test case for spending a transaction output that doesn ' t exist . 
 Fix issue with JoglInput by removing unnecessary AWT event listeners and cleaning up code . 
 Refactored List . partition to use method references and eliminated unnecessary lists . 
 Add support for logging HTTPD messages to the LogsHandler and Log classes . 
 Fix payment plugin timeout error message to include empty string fallback . 
 Adjusted comment in Main . java to correctly reflect the behavior of the XML filtering method . 
 Update LazyRequest . java to use `requestHeaders` instead of `request . headers (  ) `
 Modified LoadController . java to use mod . getModId (  )  instead of hardcoded mod id for logging and activating mods . 
 Modified FileDownloadStatus . java to add a new status  ' retry '  and adjusted isKeepAhead and isKeepFlow methods to account for it . 
 Fix ParametersList . java to correctly check for  - D followed by a property name . 
 Update license from GPL v2 to LGPL v2 . 1
 Remove unnecessary constructor and logger statement in Options class . 
 Add test case for cookies parsing supporting no value cookies . 
 Update AmazonKinesisFirehose sample code to use a configuration file for account ID . 
 Renamed TOMORROW_NIGHT_EIGHTIES to TOMORROW_NIGHT_80s in AceThemes . java . 
 Improved `GenericTypeDescription` to correctly handle null `superType` and avoid unnecessary type variable erasure . 
 Removed unnecessary import statements in background . java and basicauth . java . 
 Update Groovy version from 1 . 8 . 0 - rc - 1 to 1 . 8 . 0 - rc - 4 in GrailsCoreDependencies . java . 
 Added import statement for OStorage and modified code to use OStorageEmbedded instead of ODistributedStorage . 
 Modified the condition for adding GPX files to the list to check for lowercase file extension . 
 Remove unnecessary code and improve documentation for JavadocTagInfo and AbstractJavadocCheck
 Modified TagViewFragment to return false for hasDraggableOption when tagData is null or featured flag is set . 
 Fix issue with injection of MessageSource into GrailsHibernateDomainClassValidator . 
 Fixed invalid character check in OSchemaShared . java to allow spaces in schema names . 
 Fix vertical scrollbar position for RTL layouts . 
 Encoder :  Added loop counter to prevent infinity loop and limited loop iterations to 100 . 
 Updated JBRunnerTabs to account for hidden tabs in layout calculations . 
 Toggle notifications enabled default setting from false to true . 
 Increased the maximum number of thread executors from 1 to 5 and changed the queue from SynchronousQueue to ArrayBlockingQueue with a size of 50 . 
 Added code to handle deletion of temp directory correctly ,  regardless of whether it ' s a directory or a file . 
 Modified ActionNotFoundTransportException to include the action name as a parameter and added a constructor to set the action name . 
 Deprecated preloadTexture method and removed unnecessary code . 
Fixed issue with StringTemplateParser by adding a check for empty macro values and updated ParamTest to include a new parameter definition for FOONAME . 
 Added import statement for MessageDefinition and modified createResponseMessage method to use it . 
 Fix bug in MasterInfo . java :  change path to path . toString (  )  to prevent NPE . 
 Removed unnecessary imports and updated import statements for Model . java
 Remove protected account restriction on retweet button . 
 Updated formatting of poll (  )  method output . 
 Added error logging and debug logging for SAML attribute value processing . 
 Added import statement for android . text . TextUtils ;  and changed condition for filling in network name to account for empty strings . 
 Fixes issue with SpiceManager not stopping properly when activity dies prematurely . 
 Updated ThumbnailsCacheManager to use a drawColor method instead of a ColorDrawable to set the background color . 
 modified generateDefaultLayoutParams (  )  to return always MATCH_PARENT
 Fix print statement to not print to console . 
 Added methods getPercentageCacheMisses and getPercentageCacheHits to BdbEnvironmentStats ,  and updated the calculation of cache hits and misses . 
 Fix issue with loading DefaultNodeExtension class with two different class - loaders . 
 Fixes  # 1234 :  Changed TupleInfo . slice (  )  to use offset instead of start and end positions to calculate the slice . 
 Fixes  # 1234 :  Return non - null LocalQueueStats and fix add (  )  method . 
 Fix GroupMappingService interface to remove unnecessary exception and improve code readability . 
 Added method processInjectionBasedOnServletContext to SpringBeanAutowiringSupport ,  allowing for  @ Autowired injection based on the current root web application context as stored in the ServletContext . 
 Modified FunctionType to return true for isStructuralType (  )  if the function implements a structural interface . 
 Refactor NotificationsPendingDraftsService to use a coroutine for fetching pending drafts . 
 Add import statements for ChannelBuffers and Assert ,  and remove unnecessary import statements . 
 Added constant CANNOT_JOIN and updated tryJoinLines method in JoinRawLinesHandlerDelegate interface . 
 Added validation for GitHub token in ReleaseNotesBuilderFactory . 
 Fix issue with ApiHelper ' s handleRequest method to correctly handle HTTP request exceptions . 
 Added import statement for ArrayList and modified the getAllModules method to return a list of modules instead of a static variable . 
 Add documentation to Broadcaster interface ,  recommending suspension of AtmosphereResource before adding it to the broadcaster . 
 Fixed YAxisRenderer to handle zero interval values . 
 Disable plugin devcockpit and add lits to the list of disabled plugins . 
 Fix Eclipse classpath storage provider to update module name in project file . 
 Added import statements for FacesMessage and ConverterException ,  and updated the getAsObject method to handle NumberFormatExceptions and throw a ConverterException with a FacesMessage . 
 Fixed AnimChannel to correctly handle negative times and updated loopMode handling
 Removed unnecessary imports and abstract classes in TransportMessage ,  TransportRequest ,  and TransportResponse . 
 Fix bug in AbstractStoreClientFactory to use correct metadata key for stores . 
 Added support for displaying commit history for all branches when no specific branch is specified . 
 Updated Registry . java to include support for Java . lang . Long and updated copyright year to 2006 . 
 Rename `getCompletePredicate (  ) ` to `getHandledPredicate (  ) ` in `HandledPredicate` class . 
 Modified FMLServerStartingEvent to use MinecraftServer instead of Object and added getServer (  )  method . 
 Extend Type class to implement Iced interface and add import statement for water . Iced
 Update StackScrollAlgorithm to use state . clipTopAmount instead of child . getClipTopAmount (  ) 
 Removed unused import statements for ModelBuilder ,  Log ,  and H2ONotFoundArgumentException in JobsHandler . java . 
 Added fence call in detachFunctor to ensure pending invokeFunctor messages are processed before return . 
 Changed CalligraphyLayoutInflater ' s  ' isInline '  parameter from true to false . 
 Added createStorageDirPath method to create storage directory path with appropriate permissions . 
 Modified TransactionHandle to return implicit transaction state instead of pristine state . 
 Refactored getThumbnailForWPImageSpan to handle null context and filePath parameters . 
 Refactored OptimizingSearchHelperBase to use LanguageParserDefinitions . INSTANCE . forLanguage ( JavaLanguage . INSTANCE )  instead of StdFileTypes . JAVA . getLanguage (  )  for lexer creation . 
 Add support for JSON output in ResourceChat class . 
 Removed inherited role and user fields and added Parameters annotation to RestrictedTest constructor . 
 Refactored checkEndpoint to checkAndNudgePoints for improved accuracy . 
 Fixes issue  # 1234 :  updated jar file path to use URI instead of file :  /  /  /  to avoid issues with file permissions . 
 Updated ResourceHttpRequestHandler to use try - with - resources for closing input stream . 
 Fix HTTPServer . java to use final variable for runInternal instead of assigning a value twice . 
 Add import statement for Service in JvmComponentPlugin and PlayApplicationPlugin . 
 Modified ProxyInjector to use modId instead of modid for annotation lookup . 
 Modified PyImportElementImpl to return resolve (  )  instead of ResolveImportUtil . resolveImportElement ( this ,  QualifiedName . fromComponents ( name )  )  . 
 ReplayingDecoder :  Remove unnecessary System . out . println statement and use `newCumulationBuffer` instead of `new ChannelBuffer` for cumulation . 
 Refactor ReflectionUtilTest to include test for setting field value by ID . 
 Update BreakpointPropertiesPanel to enable condition combo when condition is empty or null . 
 Fix license reference in BaseIndex . java . 
 Update Toolbar constructor to use  @ Nullable AttributeSet attrs and remove unnecessary  ' context '  parameter . 
 Added override for recognizeArguments and isAvailable methods in GenericTypeOperationHandler and BatchEditLineHandler to improve code consistency and maintainability . 
 Remove unimplemented post method and add missing return type for ClientSideStreamProxyImpl . 
 Fixes  # 12345 :  Set valid range for presence priority and add check to prevent illegal arguments . 
 Added logging capabilities to ConnectionPoolConfig and updated toString method to include logging . 
 Fixes layout issues by adjusting the calculation of child view position . 
 Registered SerializerRegistrationsMessage . Registration class to enable receiving SerializerRegistrationsMessage . 
 fixed exception handling in HttpConnection . java by catching IOException ,  TimeoutException ,  and ClosedChannelException and rethrowing them as IOException . 
 Added method setUVRange to MeshBuilder and MeshPartBuilder to set range of texture coordinates used . 
 Add CoreAnnotations . ValueAnnotation for word / tag pairs . 
 Fix AddCallSuperQuickFix . java to handle null parameter lists . 
 Update CommittedChangesTreeBrowser to use VcsBundle and VcsDataKeys . 
 Fix SetType . java to throw an IllegalArgumentException with the correct value . 
 Refactor code to improve performance by reducing the number of unnecessary CompletionVariant objects created . 
 Remove unnecessary  '  ! sysroot . equals ( defaultSysroot )  '  check in CppConfiguration . 
 Added method setMessageId ( String )  to set the Message - ID header and store it in the mMessageId field . 
 Added VisibleForTesting annotation to TupleDomainOrcPredicate . 
 Cancel keepAliveTimeout in SearchContext ' s release method . 
 Added check for registration state before handling URI to allow for delayed registration . 
 Remove unnecessary synchronization in OperatorContext . run (  )  method . 
 Added check for null node before adding or setting item in DocumentOutlineWidget . 
 Fix ERBEvaluator to properly close input stream and handle exceptions . 
 Fixed constant folding for sparse chunks by checking for non - zero values in the array . 
 Renamed  ' blockId '  parameter to  ' blockIndex '  in TachyonFile . java to accurately reflect its purpose . 
 Added random sample data generation to SimpleActivity . 
 Fix editor height calculation to account for single - line mode . 
 Added method setPairedInfo ( String dstAddr )  to set the destination address and paired status ,  and updated loadStoredConfig (  )  and saveStoredConfig (  )  methods to be private . 
 Updated the AccessLogValve to use  % {format}t instead of  % t{format} for writing the date and time in a configurable format . 
 Committing change to remove unnecessary activeCallSessions . put ( call ,  callSession )  line as it causes memory leak and is unused . 
 Modified CallableMessageSender to use Apache License ,  Version 2 . 0 and updated package - info . java to include license information . 
 Added inspiration from the official Java tutorial to QuoteOfTheMomentClient and QuoteOfTheMomentServer . 
 Fix member accessibility check for inner classes . 
 Refactor code to improve quoted text handling ,  including adding a new showOrHideQuotedText method and moving quoted text format initialization to the constructor . 
 Add logging for rapid put / remove of SharedPreferences to prevent RuntimeException . 
 Added check for JavaxScripting availability and logging of warning message if it ' s not available . 
 Fixed search everywhere button visibility
 Added support for XML language and parser definitions . 
 Fixed issue withXmlRootElement annotation ,  now properly handling null name and empty string names . 
 Added log statement for debug logging of incoming query object . 
 Fixes bug in AppWidgetServiceImpl by changing the check for system UID to use callingUid instead of Process . myUid (  ) 
 Add test suites for ImmutableSet and ImmutableSortedSet . 
 Removed import statement for TextUtils . 
 Added method getContextProperty ( String ,  String )  and removed duplicate method with the same name . 
 Fix for issue with Tree . leftChildIndent not being updated properly . 
 Remove unused import statement and variable ,  and simplify the code . 
 Fix DataServerMessage . java close (  )  method to properly close file . 
 Removed unnecessary import statement for  ' edu . stanford . ejalbert . BrowserLauncher '  . 
Fix for ZEppelin - 234 :  HiveZeppelinDriver uses incorrect warehouse path . 
 Fixes issue  # 12345 :  Remove unused import and fix assertion message . 
 Fix code item concise identity to include method name ;  remove multiple nops before switch / array data pseudo instruction . 
 Fixed attachment saving to use file . toString (  )  instead of filename . toString (  ) 
 Fix deadlock issue by disallowing acquireExclusiveLock while holding non - exclusive lock or in a transaction . 
 Refactor AbstractMinMaxAggregation to use generics for AccumulatorStateSerializer and AccumulatorStateFactory . 
 Fix validation for UpdateRequest to disallow providing both script and doc . 
 Update SHA1 hash for Maven jar guava from d664fbfc03d2e5ce9cab2a44fb01f1d0bf9dfebeccc1a473b1f9ea31f79f6f99 to cce0823396aa693798f8882e64213b1772032b09 . 
 Add ROOT_ACCESS constant for domain admin . 
 Disable roaming for UsageStatisticsPersistenceComponent by adding roamingType  =  RoamingType . DISABLED . 
 Added method call to start Graylog server before tests . 
  Clarify that ValueGraphs are never equal to Graphs and vice versa ,  and remove unnecessary exception
 Updated copyright year to 2015 . 
 Add Euch and Neuem to exceptions list in CaseRule . 
 Add JSON mime mapping to MimeMappings . java . 
 Fixed ImagePresenter to properly handle null IDs and stream openers . 
 Added support for OMEMO fingerprints in Account class . 
 Added test for recompute executor creating a recompute plan and launching the recompute job at heartbeat . 
 Refactored IOLimiter interface ,  removing unnecessary methods and simplifying documentation . 
 Remove unnecessary System . out . println statement in UnionTest . 
 Added generic type parameter T to BloomFilter . isCompatible (  )  and putAll (  )  methods . 
 Fixed bug in PlainStreetEdge . calculateSpeed (  )  by passing correct options object
 Remove unnecessary System . out . println statement . 
 Updated import statement and added a new method to the PostfixTemplateTestCase class . 
 Update Cache and LoadingCache classes to note that while they are still annotated as Beta ,  their API is frozen and won ' t be changed without an 18 - month deprecation cycle . 
 Add field  ' currentOperationHandler '  to track the current operation handler for the thread . 
 Fix setCarSpeed method to avoid auto - generated stub . 
 fixed IOStatistics dump to correctly display avg search in hash when hashSearchRequests is 0
 Fix bug in Chronometer :  remove unnecessary log statement to prevent crash . 
 Update DeleteResponse . java to return true if a doc was found to delete ,  instead of if no doc was found . 
 Fixed progress logging to not print in quiet mode
 Added support for GridLayoutManager and updated layoutManagerType detection
 Fix AppLog to use null - safe methods for retrieving application information . 
 Add generic types to Pair return type in ParserATNSimulator . java
 Fixes  # 456 :  Throws GdxRuntimeException instead of IllegalArgumentException when allocation fails in Gdx2DPixmap constructor . 
 Added a return statement after marking incomplete messages to improve error handling . 
 Added DiagnosticGroup for duplicate variable declarations and updated TypeCheckTest to include new tests for duplicate variable declarations and static property declarations . 
Added performance testing for uploading feature . 
 Fix Wallet . java to skip over already signed inputs . 
 Modified GenerateSourcesMojo to generate BuildConfig for APKLIB dependencies only and removed unnecessary code for AAR dependencies . 
 fixed null pointer exception in transaction method by returning delegate . transaction (  )  instead of null . 
 Updated Config class to use getQueueConfig instead of getQConfig and updated import statement . 
 Fix concurrent modification exception message to correctly indicate the operation being performed . 
 Upgrade strategy for GlassFish Request :  added user principal to RequestContext . 
 Update Launcher . java to change the default extensions from  ' spl ,  org . develnext . jphp . zend . ext . ZendExtension '  to  ' spl '  . 
 Replace usage of Codehaus Plexus StringUtils with Apache Commons Lang StringUtils . 
 Refactored code to calculate offset and position correctly for variable length dimensions . 
 Refactor LookupCellRenderer to use PsiUtilBase . getName ( element )  for all elements ,  removing unnecessary checks and imports . 
 Fix ExpressionInterpreter to properly handle InterruptedException and propagate other throwables . 
 Added import statement for com . github . dreamhead . moco . HttpMethod and updated method name to  ' method '  for consistency . 
 Added handling for OutOfMemoryError in MimeUtility . getTextFromPart (  )  to prevent crashing and return null in case of error . 
 Refactor GDataImporter to use getTableRow ( scanner )  for both adding rows to project and values . 
 Modified GeneralCommandLine to add a new method appendToPathEnvVar for appending values to path - like environment variables . 
 Update FileDownloadTask to update the last task in the list when receiving a warning status . 
 Merge join operations for efficient log analysis
 Added default node property association for label d3 and edge property association for label d7 to support yEd from yworks . com . 
 Added method proxy ( Proxy proxy )  to set the request proxy and updated the Connection interface with related methods . 
 Add null check for row elements before adding them to the CSV output . 
 Added method `extendBasePanel (  ) ` to return a custom panel extension and updated the `registerValidators (  ) ` method to be nullable . 
 Fix issue with error handling in MBeanDumper . java by updating log message to include attribute name . 
 Added functionality to create parent directory for report file if it doesn ' t exist . 
 Added maxTransfers field and setter / getter methods to Request class . 
 Added constants for unsigned byte ,  short ,  and int max values . 
 Fix scaling issues :  update View ' s minimum width calculation ,  WebViewClient ' s callback parameter ,  and MediaPlayer ' s video scaling mode documentation . 
 Add LABEL family to floatFeatures map . 
 Adds methods for setting attributes for a PShape ,  including attrib ( String ,  float .  .  .  )  ,  attrib ( String ,  int .  .  .  )  ,  and attrib ( String ,  boolean .  .  .  )  . 
 Added method to return storage . isDefault (  )  in DefaultStateSerializer and added Boolean . class . isAssignableFrom ( aClass )  check in XmlSerializerImpl . 
 Update default size of IndexedRingBuffer to 128 ,  with a lower default of 8 for Android to address issue  # 1820 . 
 Update RequestServer to use the correct node index for getting logs . 
 Remove deprecated method cacheIsNotNull (  )  from PyPackageManager and PyPackageManagerImpl . 
 Refactor StableInvocationHandler to use DomManagerImpl . getDomInvocationHandler instead of Factory . createDomElementProxy . 
 Increase maxIdleTime from 1 second to 200 seconds . 
 Added error handling for empty GCM tokens when registering device tokens on remote services . 
 Added a new profile to the sample profile list . 
 Added support for null literals in string concatenation . 
 Update StorageService to correctly handle client mode and SystemTable updates
 Added ModelLoader . White . instance . register ( map )  to ForgeHooksClient . onTextureStitchedPre and removed unnecessary MinecraftForge . EVENT_BUS . register ( White . instance )  in ModelLoader constructor
 Fix debug logging to use Object [  ]  instead of string concatenation . 
 Add fine - grained color control to Palette by adding a filter . 
 Remove static final modifier from Toast variable in Toaster class
 Added `incrByFloat` and `incr` methods to `JedisCluster` class and implemented them using `JedisClusterCommand` class . 
 Added import statement for com . google . common . collect . Maps ;  and replaced ImmutableMap . Builder with Maps . uniqueIndex to create a map of property metadata . 
 Added  ' setUser ( true )  '  to TaskstJob constructor to set user flag for the job . 
 Add exception for explore . web . browser in Util . java . 
 Fixes issue with incorrect logging message format in ItemUIRegistryImpl . 
 Flush indexing stamp cache after file content change to ensure up - to - date index . 
 Added  @ SuppressWarnings ( unchecked )  annotation to MongoDbClient . java to suppress warning for unchecked cast . 
 Encoder :  Remove unnecessary import and simplify code . 
 Fixes issue  # 12345 :  Simplify conditional logic in Comment . java by removing unnecessary multiplication . 
 Fix typing handling for XmlAttributeValue . 
 Refactored HelpshiftHelper to use metadata instead of hardcoded strings for blog names and URLs . 
Added new settings for checkin server :  CHECKIN_SEND_LAST_KMSG_TIME ,  CHECKIN_SEND_APANIC_CONSOLE_TIME ,  and CHECKIN_SEND_APANIC_THREAD_TIME . 
 Added a fix to prevent exceptions when user types  ! foo in the ConsoleReader . 
 Added ProfilerConfig to MockTraceContext for proper profiling . 
 Fixes issue with wildcard type and subtype validation . 
 Removed unnecessary return statement in MembershipResource . 
 Implemented EditorImpl . createEditorScrollbarUI ( Editor editor )  to create persistent UI for scrollbars . 
 Fix hashCode method to use configs . hashCode (  )  instead of iterating over configs array . 
 Added functionality to open soft keyboard when clicking clear button in MainActivity . 
 Fix configureTarget method to use new ClosureBackedAction for better performance . 
 Modified ModuleRootManagerImpl to add a new private field myLoaded and updated loadState method to use myLoaded instead of myRootModel  !  =  null . 
 Added documentation for subclasses to throw a subclass of RuntimeException instead of returning it
 Refreshed URL and set zone count based on device info . 
 Updated MethodParameter to correctly implement equals and hashCode methods . 
 Updated constructor to accept destPointsetId instead of to ,  and changed parameter name from to to destPointsetId . 
 Fix nullability checks for ReilInstruction constructors . 
 Fix bug 54521 :  Changed DigestAuthenticator container creation to use TesterContext and TesterServletContext . 
 Added field mProperSigningKeySet to PackageKeySetData to improve signing key management . 
 Modified ProxyServlet . java to replace hyphens with dots in logger name . 
 Add getter for HttpClient instance in HttpClientTransportOverHTTP2 . 
 fixed bug in PCA algorithm by removing incorrect check for categorical data . 
 Changed import statement and added a new private static final field ourInstance  =  new IdeaTestFixtureFactoryImpl (  )  ; 
 Updated SchemaManager to correctly handle cases where a plugin doesn ' t require a database schema . 
Modified RemoteBlockInStream to use mCheckpointPos to track the position in the checkpointed file relative to the block ,  and added code to skip to the beginning of the block in the checkpoint file . 
 Add WPPrefUtils . layoutAsNumberPickerSelected ( mInputView )  ;  to set proper layout for number picker selected state . 
 Fix memory unit consistency ,  use GB instead of TB . 
 Delete LdapSettings and apply new settings to LdapAuthenticator . 
 Fix WebFilter to properly handle sticky sessions . 
 Updated SamlIdPProperties to use TimeUnit . DAYS . toMinutes ( 1 )  for cacheExpirationMinutes instead of a hardcoded value of 30 minutes . 
 Expand events are now disabled for the console reader . 
 Update Module . java and RepoParser . java to version 1 . 14 . 0 . 
 Fix issue  # 124 :  Re - write BikeFlagEncoder to use a more generic way to influence weighting . 
 Remove unnecessary import statement for Assert class from ServiceTicketImpl . java . 
 Added import statement for `PatternSyntaxException` and updated code to handle errors when compiling regular expressions . 
 Refactor SummaryPage to use HTML id instead of name for column headers . 
 Replace backToLocationImpl with getMapLayers (  )  . selectMapLayer ( view )  in MapInfoLayer ' s progressBar listener . 
 RenamingMap interface updated to implement Serializable and add a new import statement . 
 Added Touchy movement method to tvInfoMethod for expert users . 
 Added import statement for OkUrlFactory and replaced HttpURLConnection with OkUrlFactory to use OkHttpClient . 
 Extended the RelationTriple . java file to account for situations where the last chunk is the longest . 
 Added support for non - physical elements in PsiAnchor . 
 Added suffix  . bpmn to BPMN_RESOURCE_SUFFIX and added  . bpmn to DIAGRAM_SUFFIXES array . 
 Updated DashUpdatesFragment to use inflated layout and invalidate view . 
 Add hide annotation to WebHistoryItem constructor . 
 Added OnPostPopupListener to ReaderPostAdapter to handle post popup events . 
 Add persistId field to Inputs class and set it to the id of the input object in MongoDB . 
 Fix FuzzyKMeansMapper to properly handle empty clusters . 
 Fixed bug in ChunkedStream . java by correcting the offset calculation to use localReadBytes instead of readBytes . 
 Fixes issue with incorrect home button behavior ,  now uses finish (  )  instead of onBackPressed (  )  . 
 Modified TemplateDataElementType to append template tokens with custom modification . 
 Remove unused import org . omg . CORBA . MARSHAL ;  and add import com . ea . orbit . actors . annotation . IdStrategy ; 
 Add new fields to Calendar class for allDay events with local time zone . 
 Add import statement for `checkArgument` from `com . google . common . base` and use it in `CompilerContext . getVariable` method to replace `Preconditions . checkArgument` and remove unnecessary `try - catch` block . 
 Added import for ClosureCommand and modified getContext method to accept ClosureCommand parameter . 
 Added visibility check before drawing frame in ImageWallpaper
 Add import for com . thinkaurelius . titan . core . attribute
 Fixes bug in CopyStep . java where exception was not properly logged . 
 Add PATH_TO_WEB_INF_VIEWS to search paths for GroovyPageLocator . 
 Fixed placeholder handling in PicassoDrawable to prevent null pointer exceptions . 
 Enforce write access allowed check in PersistentFSImpl . close (  )  method . 
 Added OR IGNORE clause to insert statements in SeriesGuideDatabase to avoid duplicate entries . 
 Disable PyUnboundLocalVariableInspection by default
 Update ObjectFileTest to check for possibly additional line number 3 in main (  )  method . 
 Refactored commit logic to bypass local buffer for commit operations . 
 Fixed log file name to use logNameSuffix instead of hardcoded DEFAULT_LOG_SUFFIX
 Fix byte shift in ByteArrayInput . java . 
 Fix issue with import statement and replace HashMap with CaseInsensitiveKeyMap in ApplicationPushBuilder . 
 Extract hostname from HTTP Host header ,  handling cases with ports . 
 Fix MathUtils . java to use static final variable PI2 instead of calculating it on the fly . 
 Remove setting for node client . 
 Refactor NotificationsListFragment to use SupportLibrary 28 . 0 . 0 ' s LinearLayoutManager and RecyclerView instead of DefaultItemAnimator . 
 Fixes issue with ClassNotFoundException handling in loadClassForName method . 
 Updated ServerSystemTest to include new assertions for Compute Engine Database Connection . 
 Change internal implementation of Maps . of (  )  to use EnumMap instead of backing set . 
 Set titled separator text to title in IdeaTitledBorder . 
 Removed unnecessary import statement for  ' File '  and simplified code for building resource file path . 
 Added visitLines method call in NoSonarFilterLoader to handle suppress warnings
 Fixes issue with WindowManagerService returning null reference . 
 Modified ZooKeeperMasterModel to check for port conflicts after checking for job deployment to avoid misleading error messages . 
 Replace stat_playlist with feed in NavListAdapter ' s context . obtainStyledAttributes
 Added support for implicit references to tag files in JspFile . 
 Fixed JobManager . registerTaskManager to properly handle null scheduler reference . 
 Fix animation issue by specifying desired initial value for alpha animator . 
 Committed changes to LightIdeaTestCase . java and InjectedLanguageUtil . java ,  including the addition of a new method commitAllDocuments (  )  and changes to the clearCaches (  )  method . 
 Update grammar and label reference types to use predefined lexer rule properties . 
 Add selectKeys method to TableStatements class . 
 Continue running remaining configurations even if one configuration fails . 
 Updated LocalSession to use System . currentTimeMillis (  )   +  EXPIRATION_GRACE for expirationTime and added missing braces in describeCUI method . 
 Added new database fields for scaled images and updated password hashing . 
 Fixed UdpConnector to properly handle connections to localhost . 
 Add getClassSize (  )  method to DexFileReader and update AsmVerify to use it . 
 Fixed comparison of path lengths in attachRoutesForClass method
 Added spellcheck = false to text input field for search pattern to prevent erroneous spell checking suggestions . 
 Fixed bug in BusinessRuleTaskActivityBehavior by setting acceptance of rule filter correctly . 
 Added link to StoreView . html for H2O . SELF nodes in Cloud . java
 Modified DeferredType . java to implement equals method using getActualType (  )  . equals ( obj )  for better performance . 
 Remove unused import statement for org . neo4j . kernel . DefaultIdGeneratorFactory . 
 Modified TachyonFileSystemUtils to increase the default polling period for file completion status from 1 second to a generous 1 second . 
 Fix assertion message for create (  )  method in DefaultServiceRegistry . 
 Remove unnecessary import statement for  ' com . hazelcast . nio . Packet '  . 
 Add support for dropping plain text and HTML text into the editor . 
 Update DefaultConnectionProvider to set empty string for password if it ' s null . 
 Fixed issue with interrupted mid - tag in XmppConnection . 
 Enable TestUserApiAddOn . testCancelBPWthAddon and TestUserApiAddOn . testChangeBPWthAddonNonAvailable tests . 
 Added anyBoolean parameter to searches . fieldStats (  )  mock call to allow for optional query parameter . 
 Add Systrace and remove HMRClient ,  JSCSamplingProfiler ,  and DebugComponentOwnershipModule in CoreModulesPackage . 
 Added rect and ellipseMode methods to PApplet class . 
 Fixes issue with handling of unresolved methods in MethodAnnotationAggregator by changing logger . error to throw a RuntimeException . 
 Removed obsolete AJP and Coyote connectors and updated JDBC user ,  role ,  and group MBean names . 
 Fixes issue with incorrect error message in ScriptImpl . 
 Added propagation of original attributes to WebSocket message in SimpleHttpProtocol . 
 Modified JmsExchange constructor to accept Endpoint instead of JmsEndpoint and removed unnecessary import . 
 Add reflection - asm to the list of allowed API classes in the module security manager . 
 Implemented enterprise - grade FizzBuzz solution using factory and strategy pattern . 
 Remove unnecessary comment in Daemon . java
 Remove unused field TaskStatusNagger from TaskInternalServiceRegistry . 
 Fixed issue with PMML model publication by checking existence of model file before attempting to read it . 
 Fix issue with incorrect return type in DnsNameResolverContext . java . 
 Added import statement for AstridPreferences and updated showListSettingsPopover method to check for AstridPreferences . canShowPopover (  )  before displaying the popover . 
 Modified Activity . java to use savedInstanceState instead of state in the onRestoreInstanceState method . 
 Remove deprecated method onViewCreated ( View )  in HeaderTransformer and update callers to use onViewCreated ( Activity ,  View )  instead . 
 Add import statement for com . android . okhttp . OkResponseCache ;  . 
 Add LuceneTestCase . AwaitsFix annotation to IndicesLifecycleListenerTests to await a fix for a bug causing old ping responses to confuse master elections . 
 Fix audit logging to use principal name instead of principal object . 
 Commit message :  Added endReached variable to TableInputFormat to keep track of whether the end of the table has been reached . 
 Added sorting functionality to ConstantPoolTest using Arrays . sort and Comparator
 Added import statement for javax . annotation . ParametersAreNonnullByDefault ;  in package - info . java . 
 Added test case for fake failure in StringToolsTest . 
 Replace AtomicBoolean isStarted with a final field in AbstractService
 Added import statement for AuthToken and updated code to use AuthToken instead of session . getAuthToken (  )  . isAnonymous (  )  to determine if a session is anonymous . 
 Add DummyBlockReplacementTest to RegistryTestSuite ' s suite classes . 
 Added save details splitter proportion in project closed . 
 Added buffer parameter to getOutputStream method to improve performance . 
 fixed spelling mistake in mapNamesIds
 Updated DoubleConstant . SIZE to be StackSize . DOUBLE . toIncreasingSize (  )  and removed unnecessary DebuggingWrapper in ByteBuddyTest . 
 Added getContextPath (  )  method to JspCServletContext class . 
 EncodedDataType . java :  Updated the definition of the size of the encoding to reflect the product of the primitive type ' s length and its count . 
 Refactor ControllerActionTransformer to use performInjectionOnAnnotatedClass instead of performInjection . 
 Updated YammerMessagesConsumerOptionTest to include newerThan parameter . 
 Added method  ' incomingEdgeList '  to class  ' SemanticGraph '  to return a list of incoming edges for a given word . 
 Added parameter names to TargetMethodAnnotationDrivenBinder constructor
 Adds a new `equals` method to compare two PrimitiveValue objects for equivalence . 
 Change scanner table from uidTable to metaTable in TreeSync class
 Updated DaemonRegistryServices to use a cache for daemon registries . 
 Update TaskAdapter and EditTitleControlSet to use setButtonDrawable instead of setBackgroundResource for checkboxes . 
 Added support for disabled transaction isolation levels in ConnectionPageGeneral . 
Modified FileUtils . createDir (  )  to accept a path string instead of a File object and updated the method signature to reflect this change . 
 Fix for WordSplitter to correctly split words with special characters . 
 Rename class DialogFragment to RxDialogFragment and import RxDialogFragment instead of DialogFragment
 Added null check for getStructFieldData and getStructFieldsDataAsList methods in OrcStruct class . 
 Remove unnecessary hooks for clipboard manager . 
 Fixes issue with connection pooling in JedisSlotBasedConnectionHandler . 
 Fixes  # 1234 :  Removed unnecessary variable and cleaned up code for better performance . 
 Remove unused parameter `buildableContext` from method `getBuildSteps` in `UberRDotJava` class . 
 Added getters for successHandler and failureHandler in AbstractAuthenticationProcessingFilter . 
 Remove LoadBroadcaster . instance . getLoadInfo (  )  from BootStrapper . getBootstrapTokens (  )  method call
 Add import statement for AccessControlContext and modify code to use it to check permissions . 
 Added onStartTrackingTouch and onStopTrackingTouch methods to FillableLoaderPage . 
 Added error logging and handling for exception occurring during account creation . 
 Modified TachyonFileSystem to use new createEmptyFile and getOutStream methods . 
 Modified ClientService . java to return registration id for client listener and added documentation for removing listener using registration id . 
 Added import statement for NativeServices and added NativeServices . NATIVE_DIR_OVERRIDE to the list of immutable system properties . 
 Refactored loading of functions in memory to improve performance . 
 Improved optimization for match all query with  *  :  *  and  *  cases by using fieldQueryExtensions . 
 Added a new private field `maximizedPane_` to track whether the source pane is maximized ,  and set it to true when the pSourceWindowManager_ . get (  )  . maximizeSourcePaneIfNecessary (  )  is called . 
 Added alpha value setting for mCurrDrawable in DrawableContainer . 
 Added logic to show Self - Hosted Sign In form based on username / email input . 
 Implement getKey (  )  method in GeneratedMethodBinding to produce a unique key for the method . 
 Fix invalid surrogate pair tests to properly handle both high and low surrogates . 
 Added static final Pair < TokenSource ,  CharStream >  EMPTY_SOURCE and set source to EMPTY_SOURCE in constructor . 
 Updated NoopTask with new static final variables and added create method . 
 Handle case where bluetooth service proxy is already null
 Remove unnecessary destroy (  )  method from FilterBase class . 
 Update GraphHandler . java to use System . getProperty (  ' os . name '  ,   '  '  )  instead of System . getProperty (  ' os . name '  )  to fix issue with Windows detection . 
 Renamed setCamera to setLight and getCamera to getLight to use consistent naming conventions . 
 Fixes JOGL issue with null - terminated strings in glGetAttribLocation and glGetUniformLocation methods . 
 Fixes issue with incorrect handling of jar separator in JarFileSystemImpl . extractRootPath . 
 Reformatted code to update copyright year and removed unnecessary code . 
 Fix inconsistent naming convention in Crouton ' s Style . java file . 
 Added getDisplaySize (  )  method to UIDisplayElement ,  returning the display size in a Vector2i object . 
 Remove unnecessary import statement for  ' org . apache . tomcat . util . buf . CharChunk '  . 
 fixed bug in Particle . java by removing unnecessary Log . d statement and adjusting import statements . 
 fixed typo in Operator name from STING_CAST to STRING_CAST
 Modified PipeItemsWood to use InvUtils instead of Utils for getting inventory . 
 Added static modifier to private classes EmptyStateInitializer and CopyStateInitializer and changed the name of the inner class PartitionTable to be static . 
 Added Apache License ,  Version 2 . 0 ,  and updated import statements . 
 Enum TermsStatsOrder and DateHistogramInterval are being updated to use the Java 8 syntax . 
 Fix WebSocketProcessorTest to correctly close the WebSocket connection . 
 Resolve XMPP domain using DNS SRV records and display resolution process for debugging . 
 Fix issue with BOSH communication and chunked encoding in HttpBindServlet . 
 Added check to display palettes with at least the minimum number of colors . 
 Increased execution time limit from 500 to 1000 and updated port number to 7476 in ExecutionTimeLimitTest . 
 Added case for XmlTokenType . XML_ATTRIBUTE_VALUE_END_DELIMITER to XmlBraceMatcher . 
 Reverse order of siblings in ResourceBundlePropertiesInsertManagerImpl . 
 Changed syntax from kubernetes : master to kubernetes : masterUrl for KubernetesEndpoint . 
 Fix issue with DeclarationResolver throwing IllegalStateException when package is not found . 
 Refactored NotesAdapter to use a cursor loader and removed unnecessary reloadNotes (  )  call in NotificationsListFragment . 
 Disable strong rules for L - BFGS multinomial solver in GLM . 
 Added fast / dom / DeviceOrientation / basic - operation . html to the list of ignored test files due to a crash caused by the lack of a mock DeviceOrientationClient . 
 Fix bug in NettyBuffer . readBytes (  )  to ensure that the returned ByteBuf has a valid reference to the underlying buffer . 
 Remove unused import statement and simplify code . 
 Update FileWatcher to use LocalFileSystem . findFileByPath instead of myManagingFS . findRoot . 
 Updated android project files . 
 Removed import statements for Pref and SomePrefs_ . 
 Added PacketHandlerTransport to NetworkRegistry and modified GuiHandler to use new channel for energy - related packets . 
 Added JUnit4 runner and updated import statements for ConcurrencyTest
 Fix for isWritable (  )  method to use view provider ' s virtual file instead of containing file . 
 Replace `PrintStream` with `ByteArrayOutputStream` for `System . out` redirection . 
 Remove unnecessary import statement for Find in PinnedSessionTest . 
 Modified DefaultTransform . java to calculate and set destination rectangle correctly for scaling bitmaps . 
 Adds a new mechanism for injecting artifacts into Grails classes using a customizable set of injectors . 
 Added support for ShipStrategy . FORWARD in OutputEmitter . 
 Added Apache License 2 . 0 to AsyncImageView . java . 
 Remove unnecessary parameter `fs` from `remove` method . 
 Added  @ Ignore annotation to the test method testImplicit (  )  in BlueprintProduceConsumeImplicitPropertyTest . java to indicate that the test is currently failing and needs to be fixed . 
 Fix ConcurrentCloseTest to work on RoboVM by adding setAccessible ( true )  to the Method object before invoking close (  )  on the SocketChannelImpl$SocketAdapter . 
 Replace header view with expand indicator in QuickStatusBarHeader . 
 Fixes  # 615 :  Add test for completing a server stream without closing it . 
 Add support for Externalizable semantics for HyperLogLog objects without exposing public constructor or write / read methods . 
 Add import statement for ContainerUnloader and modify DefaultProcessor to use an array of ContainerUnloader objects . 
 Added frequently (  )  method to determine whether to use NoMergePolicy or NoMergeScheduler for index writer configuration . 
 Repair of graph completed in X secs . 
 Updated logger messages to use placeholder for random token generation . 
 Modified HystrixCommandProperties to use executionIsolationThreadTimeoutInMilliseconds instead of executionTimeoutInMilliseconds as a temporary workaround for compatibility with extending classes . 
 Added import statement for WindowManager and setSoftInputMode for dialog window . 
 fixed error handling for register iq
 Add import statement for javax . annotation . CheckReturnValue and annotate Defaults class with  @ CheckReturnValue . 
 Modified HttpConnection interface to add timeout parameter to shutdown method and updated close method to close all active streams before sending GOAWAY frame . 
 Enable debug logging for PassExecutorService in testCodeFoldingPassRestartsOnRegionUnfolding . 
 Added check for break characters in BitmapFont . isWhitespace (  )  and added a new method isBreakChar ( char )  to check if a character is a break character . 
 Fix empty tree check in NetworkLibrary
 Updated CallOptionsTest to use assertThat instead of manual string matching . 
 Added hasGravity method to Material enum to indicate whether a material is affected by gravity . 
 Add check for null tries before adding tries to method definition . 
 Renamed notFoundMarker to NOT_FOUND_MARKER and used it consistently throughout the code to improve readability and maintainability . 
 Adds project parameter to BinaryFileStubBuilder . buildStubTree method . 
 fixed exception handling in Main . java
 Added setupStructurePopup method to setup file structure popup in FileStructureTestBase . 
 Fix buffered description handling to prevent IOExceptions . 
 Add permission for WorldGeneratorPluginLibrary to access declared members . 
 Remove unnecessary System . out . println statement in AggregationStrategyWithPreservationTest . 
 Fix OF - 898 :  Limit number of milliseconds in RFC822 date parsing to at most three . 
 Added test for single SSL socket factory . 
 Increased connectionsPerThread from 1 to 16 in LoadBalancingProxyClient . 
Cancel long press by removing long press callback and tap callback . 
Refactored MedianOf3Number . java to use distinct pairs ,  updated NBy2PairSumToK . java to use distinct pairs and changed variable names ,  and added NotIncluding4 . java to convert base 9 numbers to base 10 . 
 Added  @ PreDestroy annotation to EmbeddedDatabase interface ' s shutdown method to enable proper shutdown of the embedded database when used in a Spring container . 
 Fixed deserialize method in StringMultiSchemeWithTopic to throw UnsupportedOperationException instead of NotImplementedException . 
 Remove unused import statement for AcceptEncodingGZIPInterceptor . 
 Removed unused  ' mOrientationChanged '  variable and updated  ' onConfigurationChanged '  and  ' onResume '  methods . 
 Added getMaxStackSize (  )  and setMaxStackSize (  )  methods to Inventory interface to retrieve and set the maximum stack size for items in an inventory . 
 Added support for simple intermediate adjective conjuncts and renamed relevant patterns and operations . 
 Replace operations . size (  )  with operationCounter . get (  )  in MemoryTranslog . size (  ) 
 Fix seek issue when seeking to a position before the media was prepared . 
 Update VCardEntry to use Im instead of Phone for TYPE_HOME and add missing Im . DATA field . 
 Added support for masking payload data when writing to the output buffer . 
 Fix bug in RelationTriple . java :  calculate longest chunk correctly . 
 Modified FMLCommonHandler to use EntityPlayerMP instead of EntityPlayer for multiplayer compatibility . 
 Modified BasicConfiguration to handle null input for getBoolOrElse method . 
 Added ordinal (  )  method call to UDPRebooted . send (  )  to fix serialization issue . 
 Extend ASTNum to match digits 0 - 9 ,  and ASTStr to match single and double quotes . 
 Remove unnecessary imports in AsyncCompletionHandler . java . 
 Added column headers for starting / ending token ,  size ,  and address to NodeProbe output
 Fix issue with alpha channel detection . 
 Update OBJLoader to set name of Node to  ' Model '  if objectName is null
 Fixed `LockHolderImpl` to use `actual` instead of `SchemaLock` in `equals` method . 
 Added imageView . setImageResource ( 0 )  to set the image resource to the placeholder image when the image is not available . 
 Increase MAX_EVENT_LENGTH from 2048 to 4096 . 
 Fix path construction in TestEphemeralFileChannel to use canonical file paths
 Enable / disable focus on comboBox based on its enabled status . 
 Rename NATS constants and headers to use CamelNats prefix
 Added null check for OTR fingerprints to avoid adding empty strings to the list . 
 Refactored OCommandExecutorSQLCreateClass to use db . command ( cmd )  . execute (  )  instead of executor . execute ( null )  . 
 Fix issue with AmazonServiceException being imported instead of AmazonClientException . 
 Fixes  # 12345 :  Dispose of children nodes when rebuilding the tree to avoid memory leaks . 
 Refactor RequestParams initialization to use HttpTask # createNewRequest instead of UriRequest #  < init >  . 
 Added check for isAdded (  )  before updating search results and empty message in TraktAddFragment and TvdbAddFragment . 
 Add check for reengagement reminders enabled preference before injecting dependencies . 
 Update license header to Apache License 2 . 0 . 
 Added ContainerUtil . addIfNotNull to prevent null pointer exceptions in getAllDescriptors . 
 Fixes  # 12345 :  Remove unnecessary CHECKSTYLE : OFF and CHECKSTYLE : ON comments . 
 Fix issue with ClassUtils import and correct usage in ObjectIdentityImpl . 
 Added import statement for ActivityTracker and updated InstalledPackagesPanel to include ActivityTracker . getInstance (  )  . inc (  )  ;  to ensure action button presentations are updated . 
 Adds a new static final string STACKTRACE_MESSAGE to GrailsConsole ,  and modifies the error message printing to include the stack trace message when the  -  - stacktrace option is used . 
 Fix OperatorTakeUntil to properly handle exceptions . 
 Fix ListCollectionsOperation to use connection instead of binding for command execution . 
Added method requestRectangleOnScreen to WebTextView to handle scrolling during zoom animation and added method inAnimateZoom to WebView to check if the web view is currently animating a zoom change . 
 Added support for async indexing in RedissonList using EVAL command . 
 Commit message :  Remove unused method getFileViewEnvironment (  )  from AbstractVcs . 
 Commits :  Update Pager to include Commits in the text output . 
 Upgrade Freemarker component ' s Configuration version from 2 . 3 . 21 to the latest . 
 Update SettingInjectorService . java to allow for dynamic hiding of settings based on location mode . 
 Added null check for imageBytes before creating an Image object . 
 Added import statement for `java . util . concurrent . RejectedExecutionException` and threw a `RejectedExecutionException` instead of logging a warning when the `socketWrapper` is null . 
 Added import statements for PyClass ,  PyElement ,  and PyFunction to enable completion for Python class and function names . 
 Added check for candidate ownership before adding it to the list of elements . 
 Fix issue with ZoomableDraweeView not correctly handling touch events ,  now returning false instead of true in onTouchEvent method . 
 Disable transactions in DBMaker . java . 
 Renamed constructor from SwipeFlingAdapterView to SwipeAdapterView and removed unnecessary parameter . 
 Added method matchingDegree ( String )  to calculate the degree of matching between a given name and a pattern ,  considering both length and capitalization . 
 Merge branch  ' feature / account - manager - future '  into  ' master '  :  Modified  ' AccountManager '  class to return  ' AccountManagerFuture '  instead of  ' AccountManagerFeature '  for better error handling and added  ' getAccountsByType '  method . 
 Remove unused import and method provideActivityContext . 
 Update FlowControlTest to use server - side SETTINGS frames
 Added new flag FLAG_PRIVILEGED to indicate that an application is permitted to hold privileged permissions
 Added a fix for a rare classloading issue in LockSupport . park ,  as described in JDK - 8074773 . 
 fixed bug in GroupMapper . java where map was being read from metadata even when value was empty
 Fix bug in Utf8Decoder :  incorrect error handling for incomplete sequences . 
 Update quarry config option to keep chunks loaded
 Remove unnecessary import statement . 
 Renamed Pact - specific constants to Stratosphere - specific constants in FileOutputFormat . java . 
 Added a ComponentAdapter to handle component resizing events . 
 Added getLoadBalancerExecutor method to LoadBalancingRxClient to provide access to the load balancer executor . 
 Disable disconnect ( addr )  method and replace with a no - op . 
 Fixed issue with empty suite not being displayed in test runner notifications . 
 Fix for KT - 27703 :  Add support for anonymous namespaces in DescriptorUtils . 
 Updated CreateFileOptions list initialization to use diamond operator . 
 Added method shouldOverrideUrlLoading to SmsSync class to handle URL clicks in the WebView . 
 Update ProducerZooKeeper to set new brokers info and close removed brokers . 
 RebalancingStore :  Add check for metadata store name in redirectingKey method to prevent redirecting to metadata store . 
 Added log message to loadMetadata method to include LoadMetadataOptions . 
 Adds a try - catch block around the onDecorateBitmap method to handle exceptions and log them using Debug . e ( e )  ;  . 
 Update option description for  -  - rewrite - labels to reflect change in label format conversion . 
 Added missing  @ Before override for AllocatorContractTest . 
 Added implementation of Serializable interface to Person and Address entities . 
 Restore local changes from stash or shelf and sort them back to their original change lists . 
 Added link to official documentation for creating swipe views with tabs using ViewPager . 
 Added method  ' withIcon (  @ DrawableRes int iconRes )  '  to class  ' ProfileDrawerItem '  for setting icon resource . 
 Added import statement for Arrays in DirectoryListingTest . java
 Fix issue with unnecessary catch block and close statement in Commands class . 
 Added support for adaptive icons by overriding onUpdate and onAppWidgetOptionsChanged methods . 
 Added import statement for Pattern ;  added field myCompiledPattern and compiled pattern in constructor ;  changed matches (  )  method to use compiled pattern . 
 Refactored RefreshableTask to use a daemon thread for asynchronous refreshes . 
 Remove unused constant MILLISECOND_PER_SECOND and private constructor . 
 Fix bug in HttpPageBufferClient where duplicate requests were not properly handled . 
 Added a check to set PREV_APPLICATION_MODE to overwrite possibly outdated value from prior follow - navigation in different profile ,  and to set AppMode based on buttons and settings . 
 Modified RILConstants . java by rearranging some request codes and removing unused ones . 
 Update SecureHttpContext to use a hardcoded subnet utils value for default context . 
 Removed import of SpreadWildcard to fix compiler error . 
 Added default bounds for virtual views and ensured client - set bounds are valid . 
 Fix issue with color index calculation for map markers . 
 Added ConcurrentHashMap for online indexes in IndexStatisticsTest . 
 Remove unused method override for GitRefManagerTest # dispose
 Implement semantical check for function calls and validate function calls on objects
 Added import statements for AppGlobals ,  Context ,  and PackageManager ,  and modified code to use AppGlobals . getInitialApplication (  )  . createPackageContext (  )  to load Chromium WebView factory class . 
 Added ignore annotation to flaky test method testGracefulShutdown . 
 Add support for resolving flexible types in ForceResolveUtil . 
 Modified VisibilityModifierCheck to account for enum constants being final by default . 
 Added assertion for applicationInfo . packageName in ResolveInfoTest . 
Added nullability annotation and return value to connectTo ( InetSocketAddress )  method . 
 Fix GitSimpleHandler . java to throw a VcsException with a more informative message . 
 Updated getHintFields (  )  to return a modifiable list instead of an unmodifiable one . 
 fixed typo in TileSourceManager . CykloatlasSourceTemplate constructor
 Remove unnecessary import statement for  ' Collections '  in JavaCommand . java . 
 Added a line to suspend animator when scrollbar value is adjusting or mouse is over thumb . 
 Added a volatile - write to the  ' running '  field in ManagementCenterService to ensure proper synchronization . 
 Modified AbstractDelegatingCasView to use  ' render view '  instead of  ' model .  .  .  '  in logger statement
 Added setup and clean methods to OrientGraphJdbcQueryTest . 
 Fix android bug :  getIntent (  )  . getExtras (  )  . getByteArray (  )  always returns null ; 
 Added import statement for  ' Collections '  and modified  ' myTagsGetter '  function to use  ' DomImplUtil . findSubTags '  method with  ' XmlTag '  parameter . 
 Fix issue with creating output directory for screen recording . 
 Added synchronized method to ensure thread safety in PythonNamesValidator . 
 Refactor TimeRange class to use Number instead of Double for relative ranges
 Fix issue caching to use clear instead of removeAll . 
 Added support for AppCompatActivity and updated toolbar setup . 
 Fixed supplicant state handling in WifiStateMachine to properly transition to DriverStartingState upon receiving INTERFACE_DISABLED state . 
 Fixes tap timeout issue by adding a check for pointer index and motion slop . 
 Fix bug in ExpandTabsReader :  handle EOF correctly . 
 Added a try - catch block to handle exceptions when testing connectivity to hosts with multiple addresses . 
 Fix FalsingManager . java to remove unused variable and remove unnecessary sessionExitpoint call . 
 Refactor AttributeRankingBuilder to use DynamicAttributeRanking for refreshMinMax . 
 Added import statement for SUSPENDED_ATMOSPHERE_RESOURCE_UUID and added method websocketResource ( AtmosphereResource r )  to Utils class . 
 Added import statement for `ProcessingException` and updated `handleInvokeException` method to handle it . 
 Renamed  ' getKind '  and  ' setKind '  to  ' getToken '  and  ' setToken '  respectively ,  and updated references to match . 
 Added documentation and changed scope of MySessionScopedBean from RequestScoped to SessionScoped . 
 Hide unused fields for unsupervised methods in SupervisedModelBuilder
 Fix bug in MetadataUtil by correcting type signature for VARCHAR . 
 Modified DebugLogs . java to update logcat command with new parameters . 
 Added import statement for org . junit . ComparisonFailure and overridden checkTestOutput method to handle ComparisonFailure exceptions . 
 Update resource roots for h2o - web and h2o - core to point to src / main / resources / www
 Fixes  # 4567 :  Handle null result and incomplete responses in AbstractClientRequest . 
 Add support for parsing TLDs with multiple URIs . 
 Update DriverPublication to use local variable for activeTermId instead of calling get (  )  method multiple times . 
 Refactor PaymentGatewayProcessor to use re - entrant plugin calls . 
 Fix issue with null pointer exception in processHandRelease method . 
 Added public access modifier to TickEvent . Type enum . 
 Adds test for move method . 
 Added import statement for GrantedAuthority and updated code to use GrantedAuthority instead of Authentication . getAuthorities (  )  [ i ]  to improve performance . 
 Added method isClosed (  )  to AsyncBatchCursor and AsyncQueryBatchCursor to check if the cursor has been closed . 
 Add a hint to avoid clipping the background of a notification if possible ,  and instead resize it to fit the screen while maintaining the aspect ratio . 
 Added test case for multi - insert with continue . 
 Update AdminServiceRequestHandler to provide more detailed error messages for file fetcher failures . 
 Added roster versioning to XmppConnection . 
Added support for sticky broadcasts in ShadowApplication . 
 Add logger for MapViewerOsmDroid and add static string for updateOpenCellIDMarkers . 
 Update ServerStarter to load lib directory from a file instead of hardcoded path . 
 Added import statement for `java . util . concurrent . TimeUnit` and modified `withPotentialTimeout` method to use `TimeUnit . MILLISECONDS` when building a `FailOnTimeout` statement . 
 Fix invalid user data exception message . 
 Fix for JoglGraphics . java to properly handle GL version 1 . x . 
 Updated SchemaManager to use  ' jiveVersion '  table instead of  ' ofVersion '  for updating version name to  ' openfire '  . 
 Added  ' public '  modifier to  ' artifactResults '  field in DependencyResolvingClasspath . java
 Enable / disable timing tests based on system property . 
 Added step counter and assertions for step counts in CucumberActivitySteps . 
 Fix JPQLSerializer to correctly handle entity names for root entities . 
 Fixed illegal state exception handling in AbstractHttp11Processor
 Add connectivity check to CarrierText to only display text when device is a phone and network is supported . 
 Simplify DeclareVariable operation by removing unnecessary Atom wrapper and returning variable value directly
 Renamed FlowScope to FlashScope and updated documentation to reflect the new behavior of maintaining variables for the next state only . 
 Added a new line of code to set the created and updated dates for invoice payments . 
 Fix MuninnPage . read (  )  to zero - fill the rest of the page instead of just up to the cache page size . 
 Updated LocalStore . java to use setHeader instead of addHeader for Message - ID . 
 Add UNSAFE . loadFence (  )  to prevent load reordering in validate (  )  and validate ( long )  methods . 
 Removed unnecessary import statement for AzureUnicastHostsProvider . 
 Added check for null before showing media controller in full screen
 Refactor VcsProjectLog to use RecursionGuard instead of RecursionManager . 
 Modified SyncManager to respect background data usage setting
 Adds a new constructor to RequestParams that allows population of multiple initial key / value string pairs and adds a new method to add a key / value string pair to the request . 
 Added code to load system properties into EventTrackerHandler ' s properties . 
 Add test for multiple underscores in domain names . 
 Modified action names to use FlushAction . NAME and RefreshAction . NAME instead of hardcoded strings . 
 Added new static final fields EXTRA_DURATION_LIMIT and EXTRA_SIZE_LIMIT to MediaStore class . 
 Update WXSDK_VERSION from 0 . 5 . 2 . 2 to 0 . 5 . 2 . 8 . 
 Fix inspection tool registration to avoid duplicates and broken factories . 
 Add TransactionalTask interface with execute method to handle transactional logic . 
 Added support for hiding action items related to the content view when the nav drawer is open
 Add DiskThresholdDecider to ClusterState . 
 Modified KdTree . XYZPoint class to use protected members instead of private ones . 
 Added check for Java identifier part in isWholeWord method to improve search functionality . 
Added Javadoc comments and updated the main method in the App class to include a prototype chain . 
 Added destroyMethod attribute to taskExecutor (  )  and taskScheduler (  )  beans to properly shut them down when the Spring application context is closed . 
 Fixed issue with ContextualUndoAdapter ' s getView method to correctly pass the contextual undo view as the third argument . 
 Remove unnecessary logging statement in SsoWebViewClient . 
 Disable on - disc stream cache for faster texture pack reloading . 
 Fix aspect ratio logic for indeterminate ProgressBar drawables . 
 Fix issue with OpenSSL socket implementation on Android . 
 Refactored subsampling layer to use  ' addiColumnVector '  instead of  ' addiRowVector '  for average pooling . 
 Increased state buffer length and adjusted offsets in LogBufferDescriptor for improved performance . 
 Add ZWave node serialization to disk . 
 Fix inconsistent type inference error messages . 
 Removed unnecessary catch block for Exception e . 
 Replace tab - separated string splitting with word - separated string splitting in ClassID . java . 
 Modified TestGremlin . java to update Gremlin script and add assert statement . 
 Modified UIServiceImpl to handle null container constraints and improved error handling for unsupported plugin components . 
 Enumerates nested files flag for recursive file enumeration . 
 Fix typo in CDFitness . java ,  correcting  ' fp '  to  ' fn '  in the definition of the equals method . 
 Fixes issue with app native libraries detection in Activity . java . 
 Added check for beta (  )   !  =  null before checking its length to prevent NPE in GLMModel . 
 Added method nfeatures (  )  to DeepLearningModelOutput to return the number of features in the output . 
 Added  ' aux '  field and getter / setter methods to LocalActor class . 
 Add method shouldShowModuleName to PsiDirectoryNode to show module name if it ' s not Cidr . 
 Remove unnecessary FeedForwardToCnnPreProcessor constructor . 
 Added try - with - resources statement to properly close InputStream and InputStreamReader in DefaultServlet . java
 Add null check for link ( id ,  siteName ,  title ,  summary ,  language ,  infos )  before adding it to myLinks . 
 Added OS . IS_WINDOWS check to skip test on Windows . 
 Added validation for label template in PipelineConfig to ensure it matches the required format ,  and added a new constant LABEL_TEMPLATE_ERROR_MESSAGE to display a more informative error message . 
 Fix resolve method to remove unused parameter `restrictToChildOrNull` and update signature to match . 
 Added PArray class to extend Expr and implement Path and CollectionType for arrays
 Fixed issue with computing languages for FindManagerImpl by removing unnecessary loop and using PsiManager . getInstance ( myProject )  instead of ProjectManager . getInstance (  )  . getOpenProjects (  ) 
 Refactor ExternalSystemAutoImporter to simplify resolveTaskState checking . 
 Fixes bug in GraphRequest :  Check for attribution identifiers before throwing FacebookException . 
 Added getExchangesCompleted (  )  method to ManagedPerformanceCounterMBean interface . 
 Added null check for Sampler object in contextBindRootScript method . 
 Fix bug  # 3826 :  Add try - catch block to handle null ref in YamlTree . getChildKeys ,  ensuring proper loading of source pane . 
 Updated keymap shortcuts for WebOpenInAction to include Shift key modifier . 
 Updated PreDexMergeStepTest to use dx command with  -  - dex flag . 
 Updated CompletionProgressIndicator to check for unit test mode before queueing updates . 
 Added logic to XmppConnectionService to handle domain lock and magic create domain . 
 Modified ModuleGraphWriter to use getIniTemplate instead of getDefaultConfig for INI template list . 
 Added support for displaying asset bundle entry name in AssetFileTreeCellRenderer . 
 Added LuceneTestCase . AwaitsFix annotation to delay allocation IT test due to a known issue . 
 Add method getDescriptor (  )  to interface InternalTaskProgressEvent . 
 Added escape key binding to interrupt R execution in TextEditingTarget . 
 Updated JBScrollPane to use ViewportBorder and removed unnecessary border creation . 
 Added field `myOriginalFile` to store the original PsiFile and updated `isUpToDate` method to check the validity of the original file . 
 Add quitUpdateThread method to notify drawTask and wait for thread to finish before quitting . 
 Fixes issue with incorrect icon ID for vibrate mode . 
 Insert BouncyCastleProvider at index 1 conditionally
 Fix incorrect decrementWindowSize argument in Http2UpgradeHandler . 
 Add serialization - related methods to the list of methods that should not be weakened . 
 Animating wallpaper crop update
 Fixed issue with backdrop animation in PhoneStatusBar by setting image drawable to null . 
 Modified TextInput method to call TextInputListener on the rendering thread . 
 Added getter method for filters in Emma class and updated EmmaMojo to use it . 
 Fix for TFS - 1234 :  Improved delete functionality to handle existing data and non - folder files . 
 Update ResponseHandler to use Request and Response instead of FullHttpRequest and FullHttpResponse
 Reset progressive start method to set mCurrentOffset to 0 and set mFinishing and mStartSection to false . 
 Added flag to show wallpaper in PhoneWindow . java . 
 Refactored unloadContent (  )  method to use WriteCommandAction for better performance . 
 Update OVERSCROLL_DISTANCE to 0 to enable overscroll for edge effects . 
 Refactor BroadcasterCacheInspectorServiceProcessor to use BroadcasterCacheInspector instead of BroadcasterCache . 
Added logging for onProgressChanged (  )  to handle unexpected progress updates
 Added `serialVersionUID` field and `Message` parameter to `ParameterException` constructor . 
 Reordered parameters in FileSystemMaster ' s getFileInfo and createFile methods to improve performance
 Fixed issue with move to top / bottom menu items not being visible when they should be
 Added PEEK_HEIGHT_AUTO constant to BottomSheetBehavior for automatic peek height detection . 
 Removed unused variable and added a new CachingDirectedGraphWalker for resolving artifacts . 
 Changed LogFormatter to check showAll or showCaughtException before falling back to caught exception . 
 Modified IQ . java to include packet extensions in the sub - element XML section . 
 Added `result` field to `ActionAnnotation` class to store the result of the annotated method . 
 Fix compiler error by adding  ' continue '  statement to skip null file processing . 
 Remove unnecessary dependency on SYSTEM module . 
 Added lookupKey field to CallerInfo and populated it from the PhoneLookup . LOOKUP_KEY column in the cursor . 
 Added flag to start indexing tours after download and scaled image to fit screen size to avoid problems . 
 Fix syntax error in BuildFileASTTest . java by replacing  '  +  '  with  '  /  /  '  in the last line . 
 Removed unnecessary imports and added import of CountDownLatch for AeronPing
 Refactored PersistitTransaction to use a thread - id / tree - name based exchange cache instead of a static map . 
 Fixes  # 12345 :  Allow requesting entity to probe presence of user . 
 Fix animation ending by adding an else statement to check if the animation has started . 
 Fix PushLog rendering by removing unnecessary border . 
 Update icon for impassible roads menu controller . 
 Replace `properties` with `propertyValues` in TestingConnectorSession . java . 
 Added setter methods for server and jdbcDriver properties in AbstractMetaDataExportMojo . 
 Refactored code to use `indentInColumns` and `indentInPixels` instead of `indent` for calculating soft wrap indent . 
 Fixes issue  # 1234 :  Corrected table name retrieval in GroupTest . 
 Added field mDate to LogEntry to store the date and time of each log entry . 
 Added import statement for XmlStringUtil and modified descriptionTemplate assignment to use its stripHtml method . 
 Add method getDefaultBlockSize (  )  to FileSystem and DistributedFileSystem ,  allowing for the optimization of large input files . 
 Updated AclExampleTest to use  ' Access Control Lists '   ( ACLs )  instead of  ' ACLs '  for clarity . 
 Fix issue with incorrect import statement in Downloader . java . 
Modified FrodoBridge . java to make startActivity and makeIntent private . 
 Updated WebService to use ` @ Override` annotations for clarity and consistency . 
 Add test case for StringUtils . replace (  )  method . 
 Added support for Apache Kafka 0 . 10 . 0 +  with new serializer and deserializer APIs . 
 Added support for PSK ciphers in the CipherSuiteConverter class . 
 Fixes issue  # 12345 :  Collapsed content view was not properly hidden . 
 Add delegation to parent role for permission checking . 
 Replace YAML parser with Jackson ' s YAML parser for improved performance and compatibility . 
 Fixed testGratefulGraphNormalized to check for ignoresSuppliedIds in addition to supportsEdgeIteration
 Add gesture detection for scrolling to MessageView . 
 Remove unnecessary imports :   ' SimpleDateFormat '  ,   ' Locale '  ,  and  ' TimeZone '  . 
 Refactored library removal to use modifiable model and removed unnecessary code . 
 Fix issue with SpeedSearchBase . manageSearchPopup method ,  ensuring that ApplicationManager . getApplication (  )  is not null and not disposed before accessing DataKeys . PROJECT . 
 Toggle handling of fields and methods in GenericExporter
 Fix proximity sensor negative events user activity . 
 Fix stubbing with subtypes . 
 Removed import statement for  ' com . hazelcast . spi . impl . RemotePropagatable '  in Operation . java . 
 Added support for custom diff handlers in FileHistoryPanelImpl . 
 Fixes issue with InputMethodManagerService ' s buildInputMethodListLocked method not properly restoring calling identity . 
 Added support for Spark 1 . 3 in SparkSqlInterpreter . 
 Fixed bug in OpenStreetMapGraphBuilderImpl by setting  ' hasBogusName '  instead of  ' bogusName '  for streets without a name or ref tag . 
 Replace GridMarkerClusterer with RadiusMarkerClusterer in CellTowerGridMarkerClusterer
 Add Android Layout and Strings files to the Important Files list . 
 Fixes issue with unsupported cancellation of ListenableFuture returned by Service . start (  )  and  . stop (  )  methods . 
