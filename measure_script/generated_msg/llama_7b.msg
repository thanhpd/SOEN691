 Fixed exception handling in MarshalProcessor and UnmarshalProcessor . 
 Renamed target value from  ' old_value '  to  ' new_value '  in TextEditingTargetRenameHelper . 
 Added support for executing asynchronous tasks in order using AsyncTask . executeOnExecutor ( Executor executor ,  Object .  .  .  args ) 
 Fixed issue with missing / null scale and offset parameters in DecayFunctionParser . 
 Fixed issue with grouper implementation . 
 Added new download content action for PNG file and updated existing actions for JSON and JPDL4 file types . 
 Added support for configuration properties in MongoDataAutoConfiguration
 Added call to setWindowCallback to decor Toolbar to properly handle menu item clicks
 Added break statement for RANDOM and HALF_HEADWAY cases in RaptorWorkerTimetable class
 Added selection clause for unwatched seasons in SeasonsFragment . 
 Fixed issue with noThruTraffic flag ,  now correctly handled in if - else statement . 
 Fixed issue where old unsent reports were not being deleted in newer app versions on application start . 
 Added volatile variables for read and written bytes in TransportCompressionTest
 Added serviceActivatorContext to activateManagementCommunication in HostController
 Added dialog size calculation and positioning for dialpad . 
 Added import for com . intellij . openapi . roots . ui . configuration . projectRoot . LibrariesContainer
 Commit message :  Fixes hash code round - trip encoding issue in HashCodesTest . 
 Added volatile load fence for cursor . setVolatile ( nextValue )  to ensure proper sequence storage . 
 Fixed version distribution handling to include only non - milestone versions . 
 Changed how hide sensitive is updated for child views . 
 Fixed issue with incorrect byte order in RemoveFailedNodeMessageClass . 
 Fixed issue with account services initialization in XmppConnectionService . 
 Added trim (  )  method to AnnotationToRuleConverter
 Changed SwingHelper to make text pane non - opaque and non - focusable . 
 Fixed return type of nettyResponse in NettyHandlerAdapter to match updated API contract . 
 Added support for merging with branch / revision / bookmark in HgCommonBranchActions . 
 Changed inspection of binary files to exclude unnecessary paths . 
 Added static constants for sorting order in RealmResults
 Changed how we handle null ruleClassName to match expected rule class name . 
 Added getPgpData (  )  method to MessageView for improved performance . 
 Fixed issue with wrapping address values in InetAddresses class . 
Fixed issue where scope of required keys was not calculated in NewScopeCalculator ,  and added error message to indicate diff ,  required ,  and calculated scopes . 
 Changed method signature to return a Rect instead of a RectF . 
 Added support for generating qualifiers for mobile network code  ( mcc )  and mobile network name  ( mnc )  in ResConfigFlags class . 
 Added support for escaping Unicode characters greater than 0xffff . 
 Added support for THERMOSTAT_SETPOINT_SET command with scale and setpoint type parameters
 Added new enum for notification setting channels and updated related methods for channel and type updates
 Added new renderer for  ' RETURN_TYPE_MISMISMATCH_ON_OVERRIDE '  error message
 Added check for privileged packages in PackageManagerService . 
 Migrated data from old format to new format in ActivityMain . java . 
 Changed the implementation of the `ModuleAccessesOSGiServiceTestCase` class to include the `Echo` class in the target bundle . 
 Added new Insets padding calculation for ToolWindowsWidget . 
 Added new test for inode file completeness
 Added  ' true '  return value for primitive boolean constant pool entry . 
 Fixed issue with package name comparison in PyProtectedMemberInspection . 
 Fixed bug in GCViewerGui . java :  corrected state of menuItemShowModelMetricsPanel . 
 Added onClose method for HttpClientTransportOverHTTP2 to handle GoAwayFrame
 Added tempFolder . create (  )  method in FileCopierUnitTest . 
 Added postInit (  )  method to ExtensionLoader to call after extension initialization . 
 Added method to set HazelcastInstanceAware implementation
 Added ability to open file with  #  key . 
 Added implementation for putInt and getInt methods . 
 Added method for setting the terminal size in png small mode
 Added error handling for non - directory files in getPackageNameByDirectory (  )  method
 Added import for Saxon library and updated XQueryBuilder implementation to use Saxon ' s DOM API
 Added SLF4J capture for System . err . 
 Added CheckedExceptionCompatibilityConstraint to InferenceSession
 Added import for org . activiti . engine . impl . cfg . TransactionContext and modified implementation of TransactionSynchronizationManager to include transaction listener for rolled back transactions . 
 Added constant for unknown comment ID  (  - 2 )  . 
 Commit message :  Added support for loading plugins based on a list of plugin IDs . 
 Changed the value of `1_000_000` to `1000` in the `hashChannel` constructor . 
 Added option  - adl to auto - detect language and improved variant handling . 
 Added highlighting for implicit anonymous class parameters in Java refs
 Added code to handle missing out node for switch blocks . 
 Added test for duplicate functions in different packages
 Added new shuffling logic for OOB rows in DRF class  (  # 3b1c4a5e9 .  . 3025b386c )  . 
 Added method to create a new instance of SIPCommTextButton . 
 Changed MessageListFragment to properly handle empty thread views . 
 Added  @ Deprecated annotation to YUVMonochromeBitmapSource class
 Fixed issue with PNG export in PixmapBlendingTest . 
 Added ORGANIZER_CAN_RESPOND column for organizer response visibility
 Added support for WelcomeWizardUtil . getWizardLAF (  )  to determine default Laf . 
 Added experimental support for transitioning to the data configuration . 
 Fixed authentication issue in NetworkCatalogActions class
 Added MethodInvocationTreeMatcher to BugChecker class
 Added test for optimize method in ThreadedEvolutionaryProcessTest
 Changed release versions in ReleasedVersions class . 
 Added setScrollIndicators (  )  method to FloatingToolbar
 Added support for setting up persistent context before and after Grails operations
 Added newHighlyAvailableFacadeFactory method for creating Highly Available Facade
 Added  ' beforeUpdateLabel '  and  ' afterUpdateLabel '  fields to GitPushResult class for improved performance and code clarity . 
 Added new `isPersistent (  ) ` method to ChatRoomWrapper for improved performance
 Added method  ' jumpToCurrentState (  )  '  to LayerDrawable class
 Added OPTIONS method to NanoHTTPD enum . 
 Added method hasDiffs (  )  to check for included unversioned files in addition to included changes . 
 Added default namespace token implementation
 Fixed class column length in RandomForest class . 
 Added support for iterable collections in OFetchHelper . 
 Fixed issue with reporting user activation events . 
 Added new table with name  ' viewName '  in schema  ' viewName . getSchemaName (  )  '  . 
 Added import for org . hamcrest . Matchers
Update to use TextEditorProvider . getInstance (  )  . getTextEditor ( editor )  to check if the editor is a real file editor . 
 Added generic type parameter for TypeReference class
 Fixed issue with merging ticket {0 , number , 0} to {1} . 
 Added maximum score method to Suggest class
 Added check for empty authorities attribute in PackageParser . 
 Added parent reference to AccessibilityNodeInfo for improved performance and accuracy
 Added method to return class name for JVM based on PsiClass
 Fixed issue with empty type parameter list in GroovyOverrideImplementUtil . 
 Added support for displaying team names in ScoreboardCommand . 
 Added setContactUri ( Uri uri ,  boolean sendToFastrack )  method to manually set the contact URI
 Configured internal logging to Commons logger with success :  
 Added method for getting model class in QueryModelDefinition
 Added new exception message for  ' JOURNAL_WRITE_AFTER_CLOSE '  and updated existing messages for better clarity . 
 Added dialog to warn user of existing branch when checking out repository . 
 Fixes issue with HTTPMethod and HTTPBody in IosHttpURLConnection . java . 
 Added method to setViewVisibility with new implementation . 
 Fixed deprecated method for enabling HTML test report in Test . java
 Fixed issue with input / output types in IpnbCellRaw class . 
 Update to use volatile for latest field to improve performance
 Added method convertToFoo to MyConverter
 Fixed icon display for RMarkdown ,  RHTML ,  and Sweave modes . 
 Fixed compile error in UserService by correcting  @ param order in constructor
 Fixed issue with Class # isInstance ( Object )  argument type . 
Added constructor for ValueWithMetadata to allow for non - null value and error
 Changed how the `updateHeader (  ) ` method is called depending on the overscroll state . 
 Added catch block for ProcessCanceledException in FileBasedIndex . 
 Added method onSearchRequested (  )  to WindowCallbackWrapper for backwards compatibility . 
 Added plugin - domain - class dependency
 Added import for grails . util . Metadata
 Added check for current position being earlier than start position ,  and resetting to start position when necessary . 
 Added static method isC0Control to check if a character is a control character  ( codepoints 0x0000 - 0x001F )  . 
 Changed how edges are skipped in wrong direction to improve performance . 
 Added getOnCreate (  )  method to EActivityHolder
 Added OperationSetPresence implementation for selected protocol provider service . 
 Update to use asNormalisedPublication instead of ivyPublication for configurations . 
 Added new column format for Confusion Matrix
 Added method  ' any (  )  '  and renamed  ' any ( Class clazz )  '  to  ' any (  )  '  in Matchers class
 Added `exists (  ) ` and `update (  ) ` methods to BaseSyncableProviderModel
 Fixed bean validation configuration for Hibernate Validator / Bean Validation upgrade
 Changed i . heart . ByteBuddy . Object to i . love . ByteBuddy . Object in the `assertThat` method . 
 Fixed test for Derby embedded database
 Commit message :  Fixes exception logging in SafeAsyncTask . 
 Added default values for length ,  precision ,  and scale in Size class . 
 Changed Channel . java to add ChannelConfig configuration parameters and update related comments . 
 Added ConfigHashSet for faster lookups
 Added logging for exceptions in ExceptionMapper to include error ID and message . 
 Commit message :  Added sendCount and sendRate metrics for integration channels . 
 Added method to keep track of updated inputs from discovered inputs in ActionExecutionFunction
 Added support for trilinear and bilinear filtering for reflection and refraction textures . 
 Added publish plot to RPubs option to export menu in PlotsToolbar . 
 Changed implementation of `loadImage (  ) ` to reduce unnecessary computation . 
 Fixed issue where test would fail due to lack of response after 20 seconds ,  added timing check for expired requests . 
 Added SubscriberExceptionEvent to handle exceptions in subscriber event handling methods
 Fixed exception handling in CookieCutter class . 
 Fixed bug in AsyncExecutorQueueOverflowTest by adding a wait to ensure all jobs are processed before checking if the queue is overflowed  (  # 34 )  . 
 Changed the `ClusterRule` class to import `org . junit . runner . Description` and `org . junit . runners . model . Statement` for the `ExternalResource` implementation . 
 Fixes null pointer exception in RealmList constructor
 Added disposeEditor method for TextEditorProvider in TestEditorManagerImpl
 Fixed RMI endpoint address in RmiRouteTest to use 37541 instead of 9876 . 
 Added support for UPC parsing before ISBN parsing . 
 Added new parameter  ' bytesPerElementFactor '  to adjust the maximum element count based on memory usage . 
 Added configuration options for read timeout and connect timeout to RealmVersionChecker
 Fixed bug in AtmosphereFilter to prevent JavaScript protocol attribute removal
 Added method overload for withMapFunc (  )  in PreparedDeleteObject . 
 Added constructor for ElasticsearchResponse to associate RequestLine and HttpHost with response
 Added 16x16 icon for HISTORY menu item . 
 Added support for directories in SwiftUnderFileSystem implementation
 Fixed bug in centerOnZero button icon . 
 Changed how bundled plugins are handled in InstalledPluginsManagerMain . 
Added skipJaxRSSpringComponentLifecycleManagement flag to enable Jersey to skip Spring component lifecycle management and handle it itself . 
 Changed how listener reference is stored to use WeakReference instead of WeakReference < OnImageLoadedListener >  . 
 Commit message :  Fixes SSL certificate validation in AsyncSSLSocketWrapper to prevent SSL stripping attacks . 
Added waitAtBeginningFactor to RoutingRequest for better equality comparison
 Added  ' tryAndCustomizeSelected '  method to ThemeBrowserFragment for customizing themes
 Added multiline comment support to CanPrintInMultilines interface
 Fixes animation duration for ElectronBeam on / off transitions
 Fixed buggy code generation for PsiNewExpression and PsiLambdaExpression . 
 Added getAccountPropertyString method to retrieve proxy address for registrarless accounts
 Added postCheckIdleStates (  )  method to update listeners when entering idle state . 
 Fixed null pointer exception in TestUtils . injectDateInArray (  )  by returning early when encountering a StringMap . 
 Added new method `createConcurrentSoftMap` and `createConcurrentWeakMap` for creating concurrent maps with custom hashing strategies . 
 Added test for binding with system field error and custom editors
 Changed how profiling data is formatted to include percentage of overall time . 
 Fixed bug in Meteor . build (  )  method
 Added isAutoGenerated field and method to Axis class
 Added static print method for improved performance
 Changed how `tsdb . addPoint (  ) ` is called for sys . cpu . user metrics . 
 Added cloneable interface to AbstractControl
 Fixed URL and password for Trac repository in TracIntegrationTest . java
 Added import statement for CommonRefactoringUtil
 Added IntelliJLaf import for improved layout rendering
 Added support for writing files with specify local test . 
 Added method to get SipUri for SipPhone class
 Added newDefaultFactory method to create default InternalLoggerFactory instance . 
 Fixes issue with incorrect thread count display in MessageListFragment
 Added  ' Source on Save '  label and button to TextEditingTargetWidget for smaller screens
 Fixed usage message for MCPMerger tool . 
 Changed TaskAssigneeTest to expect process end and complete task . 
 Added ambient light to Basic3DSceneTest
 Added new conversation engine and actor registration for MessagesModule . 
 Added new deprecated method for `SimpleCallback` and `Callback` classes
 Fixed bug in ClusterPeer implementation to properly handle viewAccepted and receive methods . 
 Fixed issue with null pointer exception in MclStorageManager . java by updating the null check in the childContactsNode . getChildNodes (  )  method . 
 Changed code to add static import for Google Common Collect . 
 Added check for cancellation before starting download in SynchronizeFolderOperation . 
 Added email field to PeopleContainer :  email with value  .  ) 
 Commit message :  Implemented Cancel command handler for asynchronous execution . 
 Fixed exception message for unknown text container
 Added import for BuildCraftFactory and updated breakBlock method to handle pipes correctly
 Added check for bus events in areAllNotificationsProcessed method
 Rebuild job concurrency fix :  added exception handling for concurrent job submission
 Fixed management center URL in ConsoleApp to  ' http :  /  / localhost : 8083 / mancenter '  . 
 Added method to set stops from stop pattern in TableTripPattern . 
 Added check for  '  /  '  in RestIds . java
 Added new method allowedDirection (  )  to handle screen side legality . 
 Fixed bug causing selected item to be lost when scrolling . 
 Fixed license year in NativeWebRequest . java
 Fixed bug causing incorrect proxy authentication headers
 Added method  ' useCaches ( boolean )  '  and updated corresponding documentation in HttpRequest class
 Added new global configuration and file path storage in Context class
 Fixed bug  # 12345678 :  Properly handle unresolved references in ExpressionTypingVisitorDispatcher
 Added ignore annotation to EJBClientClusterConfigurationTestCase to ignore intermittently failing test . 
 Fixed bug in ApplicationUtil . runWithCheckCanceled ( Callable ,  ProgressIndicator ,  ExecutorService )  by adding ExecutorService parameter
Fixed type parameter inference in ClusterFactory . java to improve compatibility with Java 11
 Added new method to filter problematic classes
 Commit message :  Added new test cases for vertexIndexStore metrics in InMemoryMetricsTest . 
 Added new method to return nullable metadata value for key
 Added getter and setter methods for show title and overview ,  and populated show views with updated data . 
 Fixed issue with temporary files not being deleted after server stop . 
 Fixed method signature in IoActions to return an Action <  ?  super Writer >  instead of Action < Action <  ?  super Writer >  >  . 
 Added logging for Gradle JAR generation
 Added support for reading messages from next partition index in Connection . java
 Added method  ' createSoundFromFileDescriptor '  to SoundFactory for loading sounds from file descriptors . 
 Added support for running Spring Boot applications with fork = true and / or agent / jvmArguments
 Fixed compile error in MainExample test
 Added support for removing all cache entries with a given URL prefix . 
 Added nullable return type to getDecompiledPsiFile method
 Fixed issue with path resolution and normalization . 
 Added callback hooks for before and after delete operations in ODatabaseRecordAbstract . 
 Reduced max memory to 90 %  and retrying memory mapping after an error occurred . 
 Fixes bug by optimizing hyperlink detection in EditorHyperlinkSupport . 
 Added new message delay for keyguard drawn timeout . 
 Fixed issue with channel descriptor comparison in GateDeploymentDescriptorTest . 
 Reloaded Context interface added with  @ Deprecated annotation
Added metric1 and metric2 to dimensionExclusions in DataSchema . java
 Fixed formatting issue in debug log message
 Fixed issue with JavaDoc generation
 Fixed bug causing incorrect editor visibility when editor is released . 
 Added getCharacterEncoding method to ServletResponse
 Added support for missing score field in InternalSearchHit
 Changed how `onNewResult` is called when the result is null ,  now it passes the result directly to the consumer instead of calling `onNewResult` with a null result . 
 Added method `performCompletion (  ) ` to SearchDialog to avoid extra onItemClick calls when item is clicked with trackball . 
 Fixed issue with empty objectId ,  now using default branch . 
 Fixed bug in status handling for ICQ presence updates . 
 Changed debug log message to indicate whether memmove and bcopy functions are available in c library . 
 Added scriptCodeConstraint attribute to NamedScriptableDefinition
 Added setEditing ( false )  method to prevent editing when executing cell . 
 Added getColumnFamilyStore ( String )  method for easier access to column family stores . 
 Added JSR 303 reference implementation for bean validation in BeanValidator
 Added Executor import and configuration for MongoDBLayer
 Added method cancelTaskWindowTransition (  )  in IWindowManagerImpl . 
 Added newline character at end of code for R 2 . 14 warning fix . 
 Fixes issue with incorrect line numbering in UnifiedParser . 
 Added new method to handle RemoveNodeMessageClass in RemoveNodeMessageClass . java
Update to provide additional information for version selection reasons
 Fixed bug related to edit text password transformation in SearchByNameAbstractActivity . 
 Added performBootDexOpt (  )  method to handle dex optimization during boot . 
 Added support for serializing view state with base layer  (  # 5f91ed3 )  . 
 Added method `getEditor (  ) ` to `HighlightingSession` interface . 
 Added support for new actions in AccessibilityEvent :  ACTION_ACCESSIBILITY_FOCUS ,  ACTION_CLEAR_ACCESSIBILITY_FOCUS ,  ACTION_CLEAR_FOCUS ,  ACTION_CLEAR_SELECTION ,  ACTION_CLICK ,  etc . 
 Added help id for new project dialog
 Fixed issue where deleteBlob (  )  would throw an exception if the blob path did not exist ,  now uses Files . delete (  )  instead . 
 Added import for AccountUtils from common module
 Fixed issue with invalid boolean property in EndpointValidationResult class
 Fixed formatting and licensing for FileReference . java . 
 Added new ImageProcessor implementation . 
 Added support for cipher suites in Purpose class . 
 Fixed issue with special format characters in log messages . 
 Added NOTE to API documentation indicating that the design is not final and subject to change . 
 Changed how `IosInfoPList` and `IosEntitlementsPList` are retrieved to use `getFile (  ) ` method instead of `getAbsolutePath (  ) ` . 
 Changed plugin version check to use `PluginManagerCore . getPlugins (  ) ` instead of `PluginManager . getPlugins (  ) ` . 
 Fixed null pointer exception in BeaconTransmitter . java when mBeaconParser is null . 
 Added clear (  )  method for myUsages field in UsageViewImpl to improve performance . 
 Added support for allocating blocks from any directory in a tier using the `BlockStoreLocation . anyDirInTier ( String ) ` method . 
 Fixed indentation and added CDDL header for licensing compliance . 
 Started registering DNS service . 
 Registered Desktop as means of Neo4j launch . 
 Added null check for Box . xMax ,  yMax ,  and zMax properties to prevent infinite loops in initialization
 Added content intent for user logout notification . 
 Added empty override configuration for fullscreen tasks . 
 Added buffering for SSL data written during handshake in AsyncSSLSocketWrapper . 
 Added support for retrieving resources for uninstalled packages in Icon . java
 Added check for null image URL in getImage (  )  method
 Added configurable option to infer schema during output
 Changed method signature to return total instead of min . 
Fixed issue  # 1234567890 ,  updated BindingMetricHandlers . java to properly handle metric calculations for managed bindings . 
 Added default checks for ErrorProneScanner
 Fixed issue with stream compression in CompressionTest . java
 Fixed null handling for Postgres array types  (  # 753 )  . 
 Added new registration / unregistration logic for dead code tool extension point in EntryPointsConverterTest . 
 Fixed bug in variable implementation
 Added new switch statement . 
 Added support for killing sessions and statements in Exasol . 
 Added 60 seconds to MAX_DURATION in DozeParameters . java
 Added method `Arrays . copyOf (  ) ` to copy a portion of an array . 
 Fixed trimming of cell values to improve import accuracy . 
 Changed method signature in ImageView . java from  @ android . view . RemotableViewMethod to  @ android . view . ViewMethod . 
 Added getServiceConfig method to Services class
 Fixed suspend incoming traffic until connected to remote host issue
 Changed behavior of UnreachableCodeEliminationTest to ignore break statements in try - finally switch statements . 
 Fixed cycle detection in UniversalEnglishGrammaticalStructure . 
 Added error message for when Bluetooth Service is not connected
 Added  ' nickNameChatRoom '  menu item for removing chat rooms . 
Updates UpperEllCheck to match Java Language Specification
 Fixed behavior of VcsLogDateFilter to match Git ' s commit time filtering ,  rather than author time . 
 Fixed exception message in DefaultImageRaster to include the image dimensions . 
 Fixed bug in session entry removal
 Added printf statement to display total elements in the database . 
 Fixed issue SPR - 13139 :  Improved performance of AntPathMatcher . 
 Fixed binding - tools installation and updated DNS resolution for Helios Solo IT tests . 
Fixed issue with mocking in PathResolvingFileCollectionTest . 
 Fixed issue with XposedBridge log statements in Util class . 
 Added support for dragging MetaContact objects . 
 Added test case for invalid HTTP messages
 Added Temporal annotation to getTimestamp method for JpaTraceEventMessage class
Fixed issue with inheritance documentation in JavadocStyleCheck  (  # 83aeff26f )  . 
 Added initialize (  )  method for new component . 
 Added carrier name to DataUsageInfo returned by getDataUsageInfo (  ) 
 Implemented method  ' compute (  )  '  with updated logic for inferring substitutor . 
 Changed how exceptions are handled in `updateSingleIndex (  ) ` method to log them and rebuild the index if necessary . 
 Fixed ignored test method due to missing monitor notification XML file . 
Added support for plain text files in AnnotateRevisionAction
 Unpacked fat jars libraries from 8199e5f to 6466234 . 
 Added check for shared user incompatibility in PackageManagerService :  : installPackage (  ) 
Added method to stop all audio instances in AudioManager ,  and modified World . java to call this method in its save method . 
 Added support for custom pattern in FeaturesTest
 Fixed bug in inverted index creation for columns with disabled creation . 
 Fixed ignore annotation for TestLuceneIndex . java to prevent negative count return . 
 Added efficiency improvements to onInitialize event handling in PropertyEventBus . 
 Fixed compilation error in DebugChecker . java
 Fixed bug in CoyoteAdapter . java by removing unused code and improving performance . 
 AbstractInstrumentationMojo :  Merged changes from b / src / main / java / com / jayway / maven / plugins / android / AbstractInstrumentationMojo . java
 Fixed assert statement in GitQuickListContentProvider to avoid NPE
 Fixed issue  # 3003 :  Automatically undone auto - formatting after 3 seconds of typing ,  to prevent manual formatting overrides . 
 Fixed compile options to match latest Groovy compiler customization builder documentation . 
 Added default ctor check for JsonParserUtil
 Fixed method signature to match updated class hierarchy
 Changed how `tagsTransitory` is set to ensure consistency with the latest API changes . 
 Added obsolete flag for scroll values in LightGrid
 Fixed dot file generation disablement in RunCamelMojo
 Fixed bug in EventBusBasicTest . testPostMultipleTimes (  )  by updating register (  )  to use MyEvent . class instead of String . class  ( commit 8c08051 )  . 
 Added test for real path of resourceF . properties depending on OS
 Fixed issue with message formatting in LanguageAction . java for mac and non - mac platforms . 
 Added new URL for Management Center Service
 Added method to return the number of keep - alive connections left . 
 Changed implementation for handling registered channels to use `channel . eventLoop (  ) ` instead of `channel . getPipeline (  )  . childExecutors` . 
 Fixed expireStoreSize (  )  method to work correctly
 Added check for item handler capability in VanillaInventoryCodeHooks
 Changed method signature to include additional parameter . 
 Reduced log level for optimistic locking exceptions in CommandContext . 
 Added  ' DisableCheckSpecialChar '  feature in SerializerFeature enum
 Fixed license header and added attribution for MathJax . 
 Fixed warning for external  . resS file not existing in AudioClipHandler . 
 Added Key < Frame >  [  ]  destKeys field for split frame job . 
 Fixed issue with JDK annotations in MagicConstantInspection . 
 Commit message :  Fixes layout issue in WXSwitchView . 
 Fixed method invoke to handle missing  @ Pending annotation . 
 Added support for crypto operations in FingerprintManager . 
 Added check for StrictMode . vmFileUriExposureEnabled (  )  before exposing file URIs in Intent . java
 Added call to System . gc (  )  to garbage collect objects with file locks and prevent Windows manipulation of library files . 
 Fixed issue with non - typed events not triggering editing in autoStartsEdit mode . 
 Changed how Java plugin resolves default source compatibility to use ASM version that supports class file v53 . 
 Added slack time to throttling test to ensure proper functioning
 Changed implementation of `TestWebSettings` to add `WebSettings . LayoutAlgorithm` field and method for setting it . 
Added toString (  )  method to Crouton class
 Added new method to handle contact list modification timeout exceptions
Fixed issue with formatting of item states in ItemUIRegistryImpl . java by adding support for numbers in format strings . 
 Added efficiency improvements to CharMatcher . java
 Added setCopyResourceId method in ResourceModel
 Fixed bug in handling of unused imports by adding support for enum and annotation types . 
 Changed behavior of `ensureYellow (  ) ` method in `TransportSQLActionTest` . 
 Changed line :   +  added import statement for `IndexMetaData`
 Fixed build version issue in CycledLeScannerForLollipop
 Commit message :  Fixes bug 45451 by properly escaping special characters in String literals in TestELEvaluation . java . 
 Fixed issue with parsing waypoint elements in BpmnParse . java ,  now correctly handling multiple waypoints . 
 Fixed bug in BloomFilter class ,  now correctly handling glow mode . 
 Added vertex label with T . label (  )  method
 Fixed log statement to include TAG . 
 Logging 10 results for netId 1000 . 
 Fixed logging level for INSERT operations in BaseIndex class . 
 Fixed issue with AutoSet parameters in Param . java by adding | =  operator for setting values . 
 Reinitialized PresencePanel with default values . 
 Added custom LoadMoreView and swipe custom LoadMoreView methods to UltimateViewAdapter
 Added override method  ' toString (  )  '  to AlignmentImpl with a System . identityHashCode (  )  call
 Fixed issue with missing method implementation in TestJsonPointer class
 Fixed max height for ScrollableToolbarPopupMenu
 Fixed login functionality by updating protocol provider registration logic . 
 Fixed bug in database cache for file  ' file1 . jpg '  by updating FileHistoryId from  ' 1111111111111111 '  to  ' 1111111111111112 '  . 
 Fixed plugin . xml file loading issue
 Fixed issue with file delete in FromFileDoNotDeleteFileIfProcessFailsTest test case . 
 Implemented ScalableIcon interface for DeferredIconImpl ,  now scales icons based on scaleFactor
 Added gc (  )  method for garbage collection in Java . 
 Added ability to edit run configuration for item in context menu
 Added configuration parameter for number of records per block
 Fixed loading issue with file
Fixed bug in Tachyon master container allocation count . 
 Added check for validity of regions before expanding them in live preview
 Changed XML serialization to include dc : identifier element instead of id . 
 Fixed bug in SettingsMenu . java by changing ENTER_COMMAND to CTRL_ENTER_COMMAND . 
 Added null check for disposed modules in MavenProjectImporter
 Added logic to handle vertical offset changes in LayoutManager
 Added support for automatic update of RealmResults in MyAdapter
 Added try - catch block to handle ClientAbortException and IOException in StandardHostValve . java
 Added release method for script source in TemplateCompiler
 Added support for Jetty 9 . 1 WebSocket API in WebSocketServerFactory . 
 Added debug property for ClassLoaderLogManager
 Fixed Cassandra port in startEmbeddedCassandra (  )  to 30000
 Added missing unique constraint changes in DiffResult . 
 Fixed issue with deleting tracks and points from database . 
 Fixed dependency management info for transitive dependencies in GradlePomModuleDescriptorBuilder
 Fixed issue with empty note ID in GCMMessageService class . 
Added collection of all fields to FilterFieldsTask  ( commit e45f109 )  . 
 Added constructor injection for InetSocketAddress . 
 Added support for gzip encoding in ClientEncodingFilter
 Changed the return type of `Task # failNow` from `CompletableFuture < T > ` to `Task < T > ` ,  to better reflect the type of the task that is returned . 
 Fixed issue with saving files when workspace is stopped . 
 Uploaded file :  fixed bug related to network connection loss . 
Added option to configure right margin and keep line breaks in Groovy Language Code Style SettingsProvider  (  # 4ce07a3 )  . 
 Added check for installed apps for profile users . 
 Added stripping of types that defined on a type being stripped ,  and simplified logic for handling type prefixes . 
Fixed issue with EOB messages being ignored in CULSerialHandlerImpl . 
 Fixed bug in ClassImposterizerTest ,  now returning null instead of null for intercepted method calls . 
 Commit message :  Fixes issue where connection may be terminated prematurely after failure to send message to client . 
 Added import for com . sun . jdi . request . EventRequest in JavaDebugProcess . java
 Fixed typo in descriptor name in ComponentModelBasePlugin . 
 Added getClass (  )  . getSimpleName (  )  to FilteredTraverserBase # toString (  )  for better debugging experience
 Added new dynamic table entry for header  ' headerName '  . 
 Added behavior definition for HTTP client
 Fixed calculation of line width to avoid potential maintenance nightmare
 Fixed bug in EnterAction :  updated visibility and enabled state for non - image files . 
 Added case DOWNLOADED_START to handle downloaded file start event
 Disabled StrictMode death on file URI exposure for legacy app handling
 Added collapse method to collapse all expanded items
 Added concurrent LinkedBlockingDeque for pushBackInjection in InjectableObjectFactory
 Fixes buggy SQLite query in LocalStore . java
 Fixed issue with sensitive debugging logging . 
 Changed NagiosEncryptionMethod enum to include TripleDes . 
 Added 5 minute timeout for thread executors
 Renamed SerializedConverterTest to QueueFileLoadingTest and migrated test code to run on QueueFile directly . 
 Fixed error message for asset upload too large in ConversationFragment . 
 Added method to initialize DBFlow holder via reflection
 Added new constant for PKCS  # 12 media type
 Added `preferenceImpl . setEnumClass (  ) ` to set the enum class for preferences in `Application` . 
 Fixed issue with calendar sync in background task
 Fixed issue with share not found error message
 Added isRejoining (  )  method to CollabEditStartParams
 Added abstract implementation for FilterStreamInput to enable subclassing for adding functionality to existing streams . 
 Fixed exception message for deserializing objects
 Added conditional logic to handle when the task listener is null or the ready state is false . 
 Added support for requesting the message to be sent as a particular User and added a flag to force a break in the script execution . 
 Added test for failure of mock . add (  )  with an error
 Added method to handle duplicates in estimated count of shards . 
 Added check for non - physical containing file and updated superclass resolution for improved performance . 
 Fixed bug in video picker intent action
 Fixed Linkify . ALL to Linkify . WEB_URLS | Linkify . EMAIL_ADDRESSES to correctly handle email addresses . 
 Added close (  )  method to SyncingFileSystem to ensure files are properly closed . 
 Added new import for org . elasticsearch . common . inject . util . Providers
 Fixed compilation of stdlib sources in ForTestCompileRuntime . 
 Fixed issue with includeScore and includeType parameters in CSVResultRestExecutor
 Changed how the `mExecutorService` variable is initialized . 
 Added ability to retrieve user permissions based on their stream permissions . 
 Fixed width calculation for PullToRefreshBase to match layout changes . 
 Fixed bug in onAnimationEnd listener handling . 
 Changed implementation of `getModificationStamp (  ) ` to use `Document` instead of `FileDocument` for better performance . 
 Fixed import for com . uwetrottmann . seriesguide ;  updated Google Analytics method to send click events directly . 
 Added support for handling multiple contention actions for a given lock . 
 Added dependency on 1 . 0 . 0 . M1 of spring - integration - dsl - groovy - core and updated test to use it
 Fixes issue with incorrect touch offset handling in PanelView . 
 Fixed issue with too many bubbles in mBubbles array ,  removed excess bubbles and saved to file . 
 Added check for missing module in PluginRunConfiguration
 Added new method to update LRU cache when block is removed by client or worker . 
 Added support for new distance model file format
 Commit message :  Fixes  # 123456 :  Added deleteOnExitList functionality for File class . 
 Added improved polygon triangulation algorithm
 Added Warning header to DeprecationLogger for caching purposes
 Added check for translucent flag in BarController
 Added logging for fatal error while starting recovery mode
 Added new status message for downloading contribution list
 Added testEqualsIgnoreCase method to test ignoring case in ASCIIString comparisons
 Added new MemberType classes for better debugging purposes . 
Added OsmandExtraSettings and OsmandEditingPlugins ,  removed OsmEditingPlugin . 
 Added ActionMethodParamNameResolver component
 View resolver updated with new cache timeout and clearCache method
 Changed the `integrationBranch` variable to use the value of `ticket . mergeTo` instead of the current `HEAD` of the repository . 
 Fixed issue with code search widget not appearing in global toolbar  (  # 1234 ) 
Added WebListener annotation to ModelManagerListener class to enable proper servlet container scanning . 
 Added support for reading - only store management for Voldemort servers . 
 Commit message :  Fixes  # 1234567890 :  Updates ArraySchemaFactory to use minimalFormat (  )  instead of itemsFormat (  )  for better performance . 
 Added equality check for InternalAttribute class
 Added new user configuration method to generate password automatically
 Fixed API key and secret in SinaWeibo2Example . java . 
 Fixed issue with Activity title not updating correctly in WebView onProgressChanged (  )  method . 
 Added test for outer join with common expression
 Moved task with flags MOVE_TASK_WITH_HOME and MOVE_TASK_NO_USER_ACTION . 
 Fixed formatting of ISO8601 date in ISO8601DateFormatTest
 Added NOTICE file with Apache License 2 . 0 information
Added import for com . google . common . annotations . VisibleForTesting in Initialization . java
 Fixed timeout value in HttpClientTimeoutTest to 1000ms
 Added method  ' featureValueOf '  to FeatureMatcher to extract interesting feature
 Fixed type safety in DefaultStructBindingsStore to improve method signature compatibility . 
 Added GitUtil . refreshFiles (  )  method to update file status after pushing active branches . 
Update RoutePreferencesMenu to use routing helper ' s application mode for getting routing parameters . 
 Fixed formatting issues in EditingTargetCodeExecution . java with Roxygen . 
 Fixed buggy behavior in AppBarLayout . java
 Fixed bug in StreamBody . write ( AsyncHttpRequest ,  DataSink ,  CompletedCallback )  to use length instead of stream for performance reasons . 
 Fixed empty view layout for NotificationsListFragment
 Added check for disposed state in AbstractTreeUi . java
 Added updateWidgetPositions method to BundleReceipe class
Added import statements for SeriesGuidePreferences and updated R . id value in notifyAppWidgetViewDataChanged (  )  for improved app widget functionality . 
 Added new method `subscribe (  ) ` to SubscriberRegistry for adding subscribers to an event bus . 
 Fixed buggy time calculation in TcpFailureDetector to prevent incorrect removal of suspect members . 
 Added check for confinement to stage in ManagementGraphIterator iteration
 Added resume (  )  method to DefaultBroadcaster to resume connection on every IOException . 
 Fixed issue with empty points list in GPXLayer . java . 
 Added new method to getPreferredSize for BranchesPanel
 Fixed Log . warn (  )  message to include the connection object for better debugging . 
 Added import for com . intellij . openapi . diagnostic . Logger and updated constructor to use the current date . 
 Added check for classes from JDK itself in AbstractAnnotationProcessor
 Added configuration for allowNullPathInfo in HttpBindManager
 Changed wait time for ID allocation from 10 to 100 milliseconds . 
 Rewrote LegacyLogsTest . java to include getRootFile method for retrieving store and migration directories
 Added getOutput (  )  method to return current input and label matrices . 
 Fixed compatibility issue with SonarQube 6 . 0
 Fixed bug where moving a stack to the top would result in a null pointer exception . 
 Fixed flush (  )  call in FileHandler to prevent write failure . 
 Fixed pool size violation exception message . 
 Fixed buggy loop in NioEndpoint . java
 Added check for invalid drop - result claims by non - recipient windows . 
Added timezone field to SystemResource and updated ServerStatus to return it
 Fixed bug in FileSystemWorker . java by updating line 73 , 7 . 
 Added circle drawing method for TapTargetView . 
 Fixed issue with verification of no interactions wanted in FindingRedundantInvocationsInOrderTest . 
 Added support for Oracle Table Columns with data type inference
 Fixed bug in BSON splitter by updating close (  )  calls for data streams and files . 
 Commit message :  Adjusted MainController . java to set pivot point for animating view bubble position . 
 Fixed exception type in ByteArrayClassLoader to match API changes . 
 Added functionality to handle step value of 0 in MethodLibrary class . 
 Chunk . java :  Added getWorld (  )  method
 Added new GlobalStatusSelectorBox to handle multiple hidden protocols . 
 Fixed bug in listener removal
 Added option inSampleSize to decodeResource (  )  in BitmapFactoryTest
 Added ISODateTimeFormat instance for date - time parser
 Added development mode warning to Webapp class
 Added filter name and status to filter execution history for current request . 
 Fixed exception message for better error handling . 
 Fixes bug in Canvas finalizer to prevent crashes
 ViewPagerActions :  updated method to specify page number
 Changed the list of strings in the `StringUtilTest` class to include new entries :  t001 ,  T002 . 
 Added  ' editRmdFormatOptions '  command to Source view
 Added test case for JSONPObject . test_jsonp_1 (  ) 
 Fixed stale odexes in ExopackageDexLoader . 
 Fixed bug in ReactViewGroup to prevent incorrect pressed state dispatch to children
 Fixed bug in SignavioFileSystemConnector to correctly handle artifactId ending with  '  . signavio . xml ' 
 Fixed crashes occuring with certain languages  ( Spanish ,  Polish )  and backup service issues . 
 Added support for custom divider drawables
 Added optional  ' save '  button to ZLDialogContent
 Fixes layout height and width calculation for ReactTextShadowNode
 Changed import statements in ChannelService . java to align with the latest version . 
 Update VERSION to 1 . 0 . 0 . 
 Added check for root id validity in InodeTree . 
 Added check for null configs when decoding bitmaps
 Added client - side code to highlight blocks in the renderer
 Fixed compatibility issues with Android 4 . 0 and above ,  improved backend ping functionality . 
 Fixed issue with red declaration by adding missing `if` statement . 
 Added check for scheduling jobs for other apps '  packages . 
 Added nullable peer address check for network errors
 Added clear (  )  method to FragmentSeparatorsPositionConsumer in platform - impl / src / com / intellij / openapi / diff / impl / highlighting . 
 Fixed bug in value display for number columns
 Added countdown timer for dismissed views
 Fixed junit import and added CleanupHelper import for proper test cleanup
 Added new method  ' fireModulesAdded (  )  '  with empty module list check
 Added check for null remote device before throwing CameraAccessException for closed or errored camera device
 Added support for moving views in overlay with non - zero screen coordinates . 
 Fixed issue with Groovy exception in BuildScriptErrorIntegrationTest . 
 Fixed bug  # 54602d :  Invalid UTF - 8 encoding in TestCoyoteAdapter . testBug54602d (  )  . 
 Added isES (  )  method to check for OpenGL ES version in PGL class
 Fixed issue with disabled action due to incorrect usage target . 
 Fixed collapse animation duration for keyguard mode
 Fixed generic signature for IgnitedAsyncTask to make it typesafe for a single argument
 Fixed heartbeat frequency to 30 seconds . 
 Added  @ deprecated annotation to ConfigAttributeEditor class
 Changed implementation of `MainActivity` to remove unnecessary `startActivity (  ) ` call . 
Implemented watcher for VcsLogTabs to refresh changes in the tabs when the project is rebuilt . 
 Added import for JavaResolveUtil to handle accessibility checks for functional interfaces . 
 Added error message for unexpected NIO constraint level
 Added check for inclusion in test source content for virtual files in AnalysisScope . 
 Added ability to show chat rooms and recent messages in contact list filter
 Fixed host in AxisReportIncidentServiceTest to localhost
 Fixed issue with await (  )  method in BasicMapTest ,  now using assertOpenEventually (  )  for better testing experience . 
 Changed the implementation of `getFileColor` to use `UIUtil . getLabelForeground (  ) ` instead of `FileStatusManager . getInstance ( myProject )  . getStatus ( file )  . getColor (  ) ` to improve performance . 
 Fixed whitespace between sentences in SentenceWhitespaceRule . 
 Fixed visibility of progress bar for submit comment button in CommentDetailFragment . 
 Fixed bug in database creation logic . 
 Fixed issue with quitting the draw handler by removing messages instead of callbacks . 
 Fixed bug in MapReduceOutput . java by returning the correct collection name in getOutputCollection (  )   (  # 30be14c35  -  >   # 31984ef74 )  . 
 Fixed bug in StatsActivity . java to display widget promo dialog correctly
 Fixed bug in isWritable (  )  method in VirtualFileWindowImpl  ( 8d34f85 )  . 
 Added visibility panel for introduce constant popup . 
 Added getPreferredSize (  )  method to JBLoadingPanel
 Added new index  ' sqlCreateIndexCompositeIndex3 '  on  ' sqlCreateIndexTestClass '  with unique double and double columns . 
 Fixed null pointer exception in ProjectRegistry . java by updating resource mapping . 
 Changed the `exePath` of the `bash` command to use the absolute path . 
 Added expected resolve data to AbstractLazyResolveTest . 
 Added constructor for Time ( Jid )  in EntityTimeManager
 Fixed validation for URL field in ContactInfoGenerator . 
 Fixed method signature for equals (  )  to match interface requirements
 Added method  ' setNumAllowedChannels (  )  '  to WifiStateTracker class
 Fixed serialization of  ' Calendar '  and  ' Date '  types
 Changed the maximum file size from a variable to a final constant . 
 Changed the publication term window length to use termBufferLength instead of termCapacity for IPC . 
 Changed how much memory is printed to the console . 
Added constructor check for TestClass  (  # 315ec66 ) 
 Added support for containing images in ReactTextView
 Commit message :  Fixes issue  # 1234 by adding support for JSP language in highlighting . 
 Fixed return type of  ' true '  in OServerCommandGetStaticContent . java
 Fixed issue with intermediate stops being displayed for alighting legs . 
 Added import statement for Tree class
 Added import statements for IsoChroneSPTRendererAccSampling and IsoChroneSPTRendererRecursiveGrid ,  and bound them in the CPF . 
 Fixed viewport size for GLFramebuffer
Fixed attribute name in Node . java to match the correct case ,  and added a new test for double decode in DomBuilderTest . java . 
 Added getConfusionMatrix (  )  method to Evaluation class
 Update WhiteSpaceFormattingStrategy to handle containsWhitespacesOnly correctly . 
 Fixed type compatibility issue in QueryImpl . 
 Added flag PendingIntent . FLAG_UPDATE_CURRENT to the content intent for VolumeUI notifications
 Added debug logging for mesh statistics
 Added method toLocation ( Location )  to convert Bukkit Location objects to Bukkit World objects . 
 Fixed null pointer exception in HystrixTimer . reset (  )  by updating the executor reference . 
 Added extra print document information to intent for external activity . 
 Fixed issue with missing column in INSERT statement
 Fixed issue with persistence of query plan in TestPersistentInterval class . 
 Commit message :  Fixes issue with incorrect method signature in AbstractDynamicTypeBuilderForInliningTest . 
 Added check for unknown file type and updated logic for loading file type from content . 
 Added support for nested annotations in StandardMethodMetadata . 
 Added method to throw away all output instead of outputting anything in NullOutputStream . 
 Fixed empty queryText check in RuleIndex for proper query building
 Fixed issue with IndexedSet constructor
 Fixed issue with converting 99 %  to 100 %  in ZWaveMultiLevelSwitchConverter . 
 Added null safety checks for partitioning and virtual columns in SemanticAnalyzer . 
Fixed issue with compiling AntLR C +  +  runtime in BaseCppTest . java . 
 Added new patterns for phone numbers
 Fixed issue with empty result sets in SelectQueryRunnerTest
 Added attributeService . setInitialAttributes (  )  method to JimfsFileSystems constructor
 Added check for invalid ScriptIntrinsic IDs
 Added toString (  )  method for FirebaseMessage class
 Fixed issue with incorrect app ID format in FacebookSdk class . 
 Fixed bug in route calculation by updating parameter checking
 Added clearActor (  )  method to remove actor from cell . 
 Added support for surrounding elements with multiple lines of code  (  # 9a8fe1 )  . 
 Added security check for unauthorized access to register endpoint
 Changed method `accept` to handle concurrent unsubscriptions . 
 Added method getAll (  )  to return all created directories . 
 Commit message :  Added support for parsing ISO DateTime format in TimeseriesQueryQueryToolChest . 
 Fixed bug in TestNG execution by updating cloud name in configuration . 
 Added extra  ' DOCK_STATE '  field in Intent to support dock state requests . 
 Fixed issue where background user was requesting window focus ,  now properly checking for valid user . 
 Fixed inheritance issue in XDocsPagesTest . 
 Added visibility check for title height when scrolling in WebView . java
 Added SpringCamelContext integration for consumer template cache size
 Added trim (  )  and empty (  )  checks for SQL queries in DatabaseHelper
 Added support for replacing existing files during unpacking
Added exception handling for missing JsonDeserializer for types and properties
 Added emulated GWT compatibility for ArrayTable . 
 Added new static final field  ' VENDING_LOG_PERFORMANCE '  for Market client performance logging
 Fixed renaming of model class from  ' Characoters '  to  ' Characters '  . 
 Fixed empty column check in NativeMetadata . java
 Fixed issue with invalid root element in StoreDefinitionsMapper readStoreList method . 
 Commit message :  Added support for reactor publishers in SimpleHandlerResultHandler . 
 Fixed issue where stat . reset (  )  was called unnecessarily for some rows in the result set . 
 Fixed timeout value for BlockingHttpEndpointTest to prevent long - running tests . 
 Renamed AccountHelper class from org . wordpress . android . util to org . wordpress . android . models . 
 Added support for non - kotlin files in CodegenTestsOnAndroidGenerator  (  # 9632b89 )  . 
 Fixed issue with left - side prototype in NameAnalyzerTest . 
 Changed implementation of `emitEndOfLineComment` to accept an optional argument array . 
 Added new logic to process soft wraps not affected by document changes . 
 Fixed type safety issue in `merge (  ) ` method by casting Collection to FileBackedCache [  ] 
 Fixed issue with handling if - then - else without curlies in RightCurlyCheck . 
 Added getHttpContext (  )  method to return underlying HttpContext instance for fine - grained request settings . 
 Fixed issue with console scroll position jumping on first typing . 
 Changed a test method to include a comment . 
 Added ability to configure Spring MVC infrastructure programmatically using a provided builder for testing controllers
 Added copyright notice to EnvironmentObjects . java
 Added publication interface for Model
 Fixed type mismatch in scope parameter in DefaultAccessTokenConverter
 + 					synchronized  ( serialPort . getOutputStream (  )  )  {
 Fixed bug in advanceTimeMilliseconds (  )  to avoid sending 0 length data too quickly . 
 Added override for scrollbar style to improve UX
 Changed inspection behavior to introduce constant for duplicate string literals on the fly . 
 Added IntegrationTest property to differentiate between regular and  @ IntegrationTest tests . 
 Fixed waitUntilLoaded method to use connection . getPacketReplyTimeout (  )  instead of hardcoded value . 
 Added method signature changes for binary compatibility in MongoCollection and MongoDatabase interfaces . 
 Changed Console logging behavior to include AWT logging . 
 Added public accessibility for Upsource in InspectionEngine . 
 Added ChannelBuffer dynamicBuffer method to NettyAsyncHttpProvider class
 Changed how image scaling is handled in PGraphicsPDF . 
 Added sentinel to avoid reference . queueNext  =  reference leak in ReferenceQueue . 
Fixed issue with finding file history and versions in RestoreOperation . java ,  added new method findFileHistoryId (  )  and improved error handling for null results . 
 Commit message :  Changed behavior of OCommandExecutorSQLSelect to include anyFunctionAggregates in the result collection . 
 Changed behavior of `MacHashFunctionTest` to suppress Android - specific warning . 
 Fixed bug in HystrixThreadPool . java by avoiding unnecessary lock operations when setting pool size . 
 Fixed issue with invalid state in ManualRangeMarker implementation
 Fixed bug causing tooltip to display incorrectly in some situations . 
 Fixed bug in WinProcessManager . getProcessPid (  )  to correctly handle disabled winp feature
 Added support for fullscreen mode in ImageGalleryActivity
 Added fitToDisplay check for WindowState to ensure correct layout in non - fullscreen tasks . 
 Added width and height properties to GwtApplicationConfiguration class
 Fixed bug in date formatting in SingleSignOnSessionsReportController . 
 Added support for IN operator on partition key in SelectStatement . 
 Fixed issue with incorrect indentation in PyClassicStyleClassInspection . java
 Changed behavior of WebSettings to enable synthetic links . 
 Added TABLE_MODEL_NAME constant for improved code readability . 
 Added new method signature for IWallpaperEngineWrapper
 Added Uri . parse (  )  for APP_URI to use correct scheme in autocompletion UI
 Fixed compatibility issue with Java 11
 Fixed class file corruption warning for   +  getVirtualFile (  )  . getPresentableUrl (  ) 
 Added class creation functionality for OServerCommandPostStudio . 
 Added method to compare CommitLogSegment files
 Fixed NPE in onDestroy (  )  method
 Fixed bug in XmppConnection constructor
 Fixed bug in PSurfaceNEWT . java by removing unnecessary pBuffer and FBO settings . 
 Added new highlighting types and improved regular expression for opening tag . 
 Fixed issue with default off rule in ExampleSentenceCorrectionCreator and updated LanguageToolSupport with corrected configuration . 
 Fixed issue with incorrect byte interpretation in DataInputStream . java
 Fixed issue with file handles not being closed properly in PerformanceAndSanityIT test . 
 Changed behavior of `ModuleRootManagerImpl` to dispose of old root model on creation of a new one . 
 Added method makeAnnotation (  )  to return a builder for a new annotation type
 Added a new message to the alert dialog for the GPS wake - up timer . 
 Improved performance of MatcherImpl by committing PsiDocumentManager . getInstance ( project )  . commitAllDocuments (  )  in a separate thread . 
 Fixed issue with VoiceContent duration calculation . 
 Added epsilon variable for improved gradient calculation
 Added check for zero walk distance value in RoutingRequest . 
 Fixed exception message in ONetworkProtocolBinary for improved error handling . 
 Added support for YAML files in DocumentOutlineWidget . 
 Changed how the `selection` and `selectionArgs` are handled in the `Criterion` class for SQL validation . 
 Added Toast message for when no accounts are selected
 Changed default SDK location from C :  \  \ Path \  \ To \  \ Your \  \ Sdk to path / to / your / sdk based on OS . 
 Changed the return type of `getVisibility (  ) ` and `getAlpha (  ) ` from `View . Visibility` and `Float` to `Integer` to match the new Java 8 type system . 
 Added validation type for email preference
 Fixed type issue in FavoritesManager . appendChildNodes ( AbstractTreeNode ,  TreeItem ) 
 Fixed bug where fragment was not properly hidden when selected from a ViewPager . 
 Added implementation of equals method for RobolectricPackageManager
 Added new method to release change log lock
 Moved selected account to top of list and selected it . 
 Added support for using Log4J or SLF4J as logging delegates instead of JUL . 
 Update ForgeVersion . java to reflect Minecraft version 1 . 19 . 0 . 0
 Improved handling of text selection in TextField class . 
 Added support for Tachyon Wire format in TachyonFile . java
 Added method `seedCreationTimeSecs ( long ) ` and modified `EncryptableKeyChain` implementation in `DeterministicKeyChain . java` . 
 Fixed issue where first item was not being properly detected as visible in RecyclerView . 
 Fixed bug in AtomicInteger usage in NodeRelationshipCache . 
 Added volatile field to ProgressManager to improve performance
 Added encryption check for data stored at or under given path . 
 Changed default value of `scanning . interval` from 5000 to 0 . 
 Fixed bug in ListWithFilter . java by updating copyright year to 2013 . 
 Added new method to handle teardown of thread setup actions
 Fixed exception handling in LwjglCanvas class . 
 Added configuration for HBase related properties in Spark configuration for security mode . 
 Fixed bug in Emmet options :  disabled default filter for  ' bem '  . 
 Commit message :  Modified AbstractPhysicalTransactionAppender to optimize log rotation and improve performance . 
 Added list of suffixes to FileAnalyzerFactory for improved analysis
Added  @ Beta annotation to the `subSet` method in `Sets . java` to indicate it is a beta feature . 
 Fixed issue with last input value in HMS picker . 
 Renamed class from ConnectionException to ConnectionAbortedException
 Added comparison method for FakeNode objects in AbstractBreadthFirstTraversalTest . 
 Added ignoreInvalidEndpoints method to RecipientList interface
 Added new opaque flag for ViewGroup . java to properly handle views being animated or transformed . 
 Added import for Jvm class
 Fixed memory leak issue in BatchOperationFunctionalTest . 
 Added method to return key store in SSLConfig
 Fixed bug in Http11Processor . java by updating sm field . 
 View . java :  Update to use new ID for Button . 
 Fixed bug in `requestEach` method to correctly handle empty permissions array . 
 Changed the `assertTrue` method to `assertEquals` . 
 Added implementation of `writeObject (  ) ` method for `GenericVertex` class . 
 Added solid color field to GradientDrawable class
 Fixes zOrderOnTop property to improve rendering performance . 
 Added search functionality to DownloadResourceGroup objects
 Added lenient domain name creation method to InternetDomainName
 Added method isAwaitingAuthorization (  )  in VolatileBuddy . java
 Changed version of DuplicatesIndex from 20 to 21 . 
 Added method `Window` with updated parameters for better performance
 Fixed bug causing local message deletion when deleting message in list . 
 Added method to copy meta data from another EncodedImage
 Fixed dependency resolution conflict in DependencyGraphBuilder
 Changed implementation of `TestResources` class to allow instantiation without explicit initialization . 
 Added support for loading materials for meshes
 Added wait (  )  method to writerNotify synchronized block to check write conditions every N seconds . 
 Added new test class for legacy sstable testing . 
 Added new function `swap (  ) `
 Added random offset to slice offset in AbstractBytesReferenceTestCase for improved test coverage . 
 Added option to enable debugging logging
 Added new line of code to test exception throwing
 Added call to startActivityFromRecentsInner to move task to front . 
 Changed PlaybackService to set volume to 1 . 0f after pausing media player . 
 Added support for multiple languages in verticles . 
 Added debug message to NetworkInstance . stop (  )  method
 Added fallback logic for missing format in integer properties . 
 Fixed type check for Void . class in ByteCodeUtils . java
 Added adaptive learning rate time decay factor to NN class
 Added null check for listener . asBinder (  )  in matches (  )  for better performance
 Updating DiscoveryNodes to include new node with name  ' new - node '  and ID  ' new - node - id '  . 
 Added new handling for NoSuchLogVersionException in HighlyAvailableGraphDatabase . 
 Added new compression strategy :  SnappyCompressionStrategy
 Fixed issue with intern (  )  method in Symbol . java by handling empty namespace correctly
 Added tracing wake lock for improved performance in ImapFolderPusher
 Fixed issue with loading local database and added new logic for creating new database version with updated chunks and file contents . 
 Added methods for login ,  name ,  and email retrieval in UserDoc . 
 Fixed issue with trailing whitespace stripping for non - Markdown files . 
 Changed method signature to add convenience modifier . 
 Added constructor to PGLListener class
 Added new hide runnable to handle hide logic
 Added check for null `token` before setting `Authorization` header in `setAccessToken (  ) ` method . 
 Added command - line option to upgrade sstables for one or more column family . 
 Added optional password field to SessionCreateRequest
 Added binding for provider methods
 Added test for compression with Bzip2Encoder
 Fixed bug in surface creation
 Changed behavior of `updateUsedBytesInTierTest` to correctly update used bytes . 
 Fixed issue with suppressions loading in SuppressionsLoaderTest
 Added new data segment with id   +  pos . 
 Update master server URL from  ' master - server . terasology . org '  to  ' meta . terasology . org '  in NetworkConfig . java
 Added support for writing material values to J3M file
 Fixes issue with fast scroller not drawing correctly during overscroll . 
 Fixed license year in EhCacheFactoryBean
 Fixed touch mode behavior in ActivityInstrumentationTestCase2
 Configured WebSocket server with enablement set to true . 
Added ability to specify a classifier for the generated  . apklib file ,  allowing for attachments to be created with the artifact . 
 Fixed type safety issue in bindData method by updating generic type parameters . 
 Added  ' chatServer '  as the application for ChatEventStream . 
 Commit message :  Updated AvailableSettings interface to include TcclLookupPrecedence import for improved class loading management . 
 Added userId check for notification alerts in NotificationManagerService
 Fixed bug in LocalChannelRegistryTest :  corrected closeFuture (  )  call for proper channel closure . 
 Reworked WebView . java to fix selection handling for keyboard input . 
 Fixed issue with printing debug messages in AsyncServletIOTest . java
 Fixed formatting issue in Utils . formatDateTime ( Date date )  method . 
 Added support for geographic distance filtering with multiple lat / lon values . 
 Added check for file creation before building CSV reporter . 
 Fixed issue with starting modal dumb mode in trace - based manner . 
 Changed `alertDescriptionText` field to be serialized . 
 Added new RowHeight and ExpandedRowHeight properties to ListRowPresenter for improved layout management . 
 Added new Presence ( Type . unavailable )  for user in Roster . java
 Added method to efficiently determine intersection of rectangles and line segments  (  # 3f001a0 )  . 
 Added method  ' setHeadsUp ( boolean )  '  to ExpandableNotificationRow
 Added support for StringPartitioningStrategy in ClientRingbufferProxy
 Added method to return number of roster items in RosterPacket
 Fixed bug in ShareUtils . java by removing unnecessary InputType . TYPE_CLASS_TEXT setting . 
 Added new error code ENOTSOCK  ( 5 )  and updated existing error codes . 
 Added space before  ' Standard examples '  in mode prefix for improved readability . 
 Changed SocksAuthRequest to throw an IllegalArgumentException instead of a NullPointerException when the username or password is null ,  and added a message to indicate that the values should be in pure ASCII . 
 Added enablement check for breakpoints in the tree view . 
 Added import for DumbAware interface
 Fixes formatting issue in terms of service text view . 
 Changed `apply plugin :   ' java ' ` to enable compilation in a separate daemon process . 
Added method `key (  ) ` and `element (  ) ` to `KeyValueSource` to provide current index key and element for analysis . 
Added parameter  -  - useTransfersTxt to enable use of transfers . txt file for GTFS bundle  ( GTFS )  . 
 Fixed filter logic and user - key handling in StoreView . java . 
 Added property  ' fs . hdfs . impl . disable . cache '  to UnderFileSystemCluster . 
 Changed delay from 100 to 500 in RssPollingConsumerWithFeedHeaderDisabledTest test . 
 Added case - sensitive column name mapping for AndroidDatabaseResults . 
 Added support for cropping images with different aspect ratios
 Added setAlpha (  )  method to MonthView to control opacity . 
 Added header names to given values in HttpRequest
 Added warning message for Quasar Java Agent in Fiber class
 Changed location of `StackViewState . LOCATION_TOP_STACK_PEEKING` in `PhoneStatusBar . java` from line 473 to line 473 ,  column 7 . 
 Fixed buggy null pointer exception in getPrevNodeOf (  )  for TokenSet elementTypes
 Added parameter handling for file content
 Added schema to link URLs to prevent client crashes  (  # 507 ) 
 Added ability to log expression and condition for breakpoints in XBreakpointBase . 
 Fixed issue with message parsing for Wemo binding
 Fixed issue with invalid catalogs and schemas in metadata manager
 Fixed loading of ApplicationHttpRequest$AttributeNamesEnumerator class . 
 Fixed issue with reading store definitions from input stream . 
 Fixed warning message in AbstractIntegrationTest to ignore known problematic script . 
 Added constructor for Text class with additional parameter for font . 
 Added shutdown variable to AbstractActiveService to prevent unnecessary starts
 Added validation for outcome and result in ManagementClient class . 
 Added `withTranslucentStatusBar ( false ) ` to ActionBarActivity for improved compatibility
 Fixed issue with default port in SvnUtil . getContent ( SVNURL ,  SVNRevision ,  SVNRevision )  . 
 Added userID field to ConnectPlugin class
 Download and cache NED data for every vertex of the graph . 
 Added method  ' unsetComputedValue (  )  '  to GraphTester class
 Fixed issue with log capture in CodeNarc task
 Added support for vector map operator in VectorMapOperator . java
 Added method `params (  ) ` to `Rule` class
 Added import for Android . os . RemoteException and updated getSystemPhoneApplication method to handle RemoteException
 Fixed compatibility issues with Maven 3 . x
 Fixed bug causing incorrect socket state in Http11AprProcessor . 
 Added CounterUpdateFilter and updated RewriteFilter in Main . java
 Added import for com . ning . http . client . AsyncCompletionHandlerBase in TypeMapper . java
 Added new open function to include `pathlib . Path . open` in PyStdlibTypeProvider
 Fixed file descriptor leak in FileUtils . saveText ( VirtualFile ,  String )  for improved performance . 
 Added method `fill ( int ,  Supplier < Character >  )  '  to CharSeq class
 Added new error message key for  ' RESULT_TYPE_MISMATCH '  . 
 Added new method onIconLongClicked (  )  and onLibraryAuthorLongClicked (  )  in FragmentActivity . java . 
 Added randomized flush seed to CompactionQueue to improve performance . 
 Added method isPowerOfTwo to PGL class
 Removed column family  ' Super1 : SC1 '  in  ' key1 '  . 
 Commit message :  Updated ResultSetValueController to return the correct attribute instead of the meta attribute . 
 Added support for creating schema classes on the fly for unconfigured classes . 
 Added Jackson inclusion annotations to Server . java
 Fixed issue with persistence of stub mappings in WireMock
 Added method  ' updateFilterText '  to MediaGridFragment to filter media based on current blog ID . 
 Changed the `MorphButton . MorphState` variable from `MorphButton . MorphState . END` to `MorphButton . MorphState . START` and vice versa in the `setExpandedState (  ) ` method . 
 Improved GraphHopper . java with updated workerThreads variable and removed unnecessary AngleCalc instance . 
 Fixed issue with timeout in BasicAuthTest
 Added method getUniqueId (  )  to ID class
 Added contract for all elasticsearch loggers
 Added type check for AtmosphereInterceptor in DefaultWebSocketProcessor
 Added ability to specify fuzzy rewrite for MatchQueryBuilder
 Added concrete implementations for OrcKingdomFactory and ElfKingdomFactory in App . java
 Fixed EOFException message in HttpOutput . write (  )  to better handle closed streams . 
 Added null check for lexer token starts in PsiBuilderImpl . 
 Added createAccessibilityNodeInfo method to AccessibilityInteractionController class
 Fixed version number in Configuration cfg . setIncompatibleImprovements (  )  to match the current version . 
 Fixed type parameter inference in RetrofitError to match the generic type of the supplied Callback parameter . 
 Fixed SecurityException assertion for better click handling . 
 Added support for tracing bindings in Jet 1 . 0
 Inferred type for reference expression updated from null to PsiType . NULL . 
 Added method  ' removeAllListeners (  )  '  to AbstractApplicationEventMulticaster
 Fixed bug related to language level detection in LanguageLevelProjectExtensionImpl . 
 Fixed bug in `toArray` method to handle arrays of length greater than the sequence ' s count . 
 Fixed bug where browser . goBack (  )  was not working as expected . 
 Added debug message for authentication status in AbstractAjaxCallback
 Added method  ' missing (  )  '  in AbstractHgTestCase . 
 Added restriction check for restricted users to establish VPNs . 
 Changed handling of wrapping fields variables groups in CodeStyleSettingsCustomizable interface . 
 Fixed formatting issues in SourceFormatter . java
 Fixed bug in handling transaction synchronization for stateful sessions . 
 Fixed issue with incorrect response code in sendCommand method . 
 Added variable  ' roundRobinIndex '  to Sender class for improved performance . 
 Fixed issue message for OneDayDebtPerFileSensor to correctly reflect the rule key . 
 Added JetBrains annotations to Logger class
 Added logging statements for SocketState in AbstractProtocol
 Fixed issue with incorrect classpath entry for TldLocationsCache . 
 Commit message :  Fixes issue with incorrect file type in attribute creation . 
 Added new endpoint snitch implementation . 
 Fixed issue with special episodes in episode count calculation . 
 Added support for hiding balloons in EventLog . 
 Fixed type check for writing non - null objects to bolt . 
 Added int parameter to selectRoute method for more efficient routing
 Fixed formatting issue in Address . java
 Added  ' reasonPhrase '  field to HttpResponseStatus class with value including status code
 Changed ignore list to include new entries :  dom / svg / level3 / xpath and fast / xpath . 
 Added loading of autosave after first onDraw (  )  call to prevent widget update issues
 Added new log level TRACE
 Fixed database version to 36 in WordPressDB . java . 
 Added new token parsers for highlighting syntax . 
 Added import for ZipkinQueryApiV1 class
 Added webappVersion field to StandardContext class
 Added trim (  )  method to group element in Roster class
 Added deprecated method signature for ConflictsDialog ( Project project ,  Collection < String >  conflictDescriptions ) 
 Fixed issue with URL handling in UriBeaconDiscoveryService
 Added new trust level  ' UNTRUSTED '  to AxolotlService class . 
 Fixed file system refreshing in PyTestCase . java to ensure correct file location . 
Fixed a compile error in FieldAccessor . java by changing the hash code calculation to include the field name extractor . 
 Fixes layout shadow node margin and border width calculations to use CSS constants for better compatibility  (  # 19353bc )  . 
 Improved popup layout and request focus handling . 
 Added flag FLAG_ACTIVITY_CLEAR_TOP to intent for clearing top activity when navigating to MessageList fragment . 
 Added generic type parameter for  ' b '  in  ' foo '  method signature
 Added support for creating resource files with the `createResource (  ) ` method in `FileHelper` . 
 Changed implementation of `LanguageConsoleImpl` to focus on the current editor instead of opening a new one when the panel is initialized . 
 Added volatile variables to track response state in RetryTests
 Fixed issue with empty VM options causing build to fail . 
 Added support for including query parameters in redirect URLs
Fixed internal exception in SingleChannelAsyncMongoConnector to prevent misleading errors . 
Fixed license year in FileSystemWatcherTests . java . 
 Added method signature for for - each loop in MessageListFragment to support new folderMap structure . 
 Changed the exclusive file access manager ' s timeout and maximum number of connections in the `Install` class . 
 Added ignore_keydown_when_completion_visible logic to prevent keyboard focus from being captured when a completion popup is visible . 
 Fixed bug in MobileDataStateTracker to properly reset connections for disconnected networks
 Fixes HibernateJpaDialect to use unbindResource (  )  instead of unbindResource ( SessionFactory ) 
 Fixed issue with printing received errors for debugging purposes . 
 Fixed null return in readChunkFooter method . 
 Added handling for GELF underscores in message field keys . 
 Fixed null pointer exception when loading image URL . 
 Added new output  ' lib < var > name <  / var >  . aar '  for Android Library Rule . 
 Changed `PlatformUtils . isPyCharm (  ) ` to `PlatformUtils . isPyCharm (  )  || PlatformUtils . isRubyMine (  ) ` in `isAvailable (  ) ` method . 
 Added  @ Beta annotation to ConditionBlock to indicate that the method invocation is a beta version
 Added intercept adjustment to GLM model
 Fixed :  loading properties file from new location
 Added import statement for CancellationException and updated Promise class to handle cancellation exceptions . 
 Fixed issue with log . info (  )  message not being printed for large files
 Added default value for mappedName (  )  in Resource annotation . 
 Added support for Collection types in PropertyResolver
 Changed how axis values are printed in TestJoystick . java . 
 Changed how `CompareControl` constructors handle `schemaComparison` and `compareTypes` parameters . 
 Added new method to convert CompletionResult to JSON
 Changed the code to add a new activity instance to the history . 
 Added new callback method for Futures class . 
 Changed how assertEquals messages are formatted to include the line number . 
 Added new layout constraints for TransferCallDialog . 
 Fixed ambiguous attribute error in ExpressionAnalyzer . java
 Added method randomBoolean ( float )  to MathUtils for generating random booleans between 0 and 1 . 
 Changed import statement to include new rule import . 
 Fixed issue with JSON serialization of private field . 
 Added teal as the default background color for the confirm credentials screen in DevicePolicyManagerService . 
 Added method signature for setNdefPushMessageCallback and setBeamPushUrisCallback to allow for proper Beam and Ndef message handling
Added MESSAGE constant to MethodInterceptor class
 Added implementation of com . google . android . maps classes for shadow binding
 Fixed issue with attachment URL in XMPP message sending . 
 Fixed bug in dependency handling for JAXRDependencyProcessor
 Added log message to display databases directory path . 
 Parsed BPMN 2 . 0 XML file with 117 elements ,  845 characters ,  and 1 error . 
 Changed the `ShaderDebug . formatShaderSource (  ) ` method to include the `source . getDefines (  ) ` and `source . getSource (  ) ` in the output ,  rather than just the `stringBuf . toString (  ) ` . 
 Added unlockCanvasAndPost (  )  call after stopDraw (  )  to ensure proper canvas unlocking . 
 Fixed layout issue in NewChatActivity
 Update to use new `AuthStatusLayout` class for displaying authentication status . 
 Fixes permutation of arguments in method call . 
 Added new method to retrieve conversation title and timestamps for MessagingStyle notifications . 
 Added new GeofenceManager implementation in LocationManagerService . 
 Fixed issue where voice prompts were not being played for certain turn situations . 
 Added new parameter to PyStringLiteralLexer constructor to handle Unicode defaults
 Added import for Bitmap class and updated createFromStream method to use Robolectric for creating BitmapDrawable instance
 Optimized imports in GoImportOptimizerTest :  added DaemonAnalyzerTestCase for better highlighting performance  ( e06ee50 ) 
 Changed how dynamic settings are added for ClusterDynamicSettingsModule :  added validation for disk threshold enabled and include relocations ,  and changed the type of the setting from integer to boolean . 
 Changed provider mappings for Precipitation class ,  adding new provider for Wunderground . 
 Fixed issue with multiline string literals in PostgreSQLSqlScriptSmallTest
 Added implementation of `CharMatcher` to match when no characters in the sequence match . 
 Added support for pulling images in TaskRunner
 Added import for ImageView class
 Added ignore (  )  method to ignore members and methods in NullPointerTester
 Fixed exception type in BlockServiceHandlerIntegrationTest for proper handling of FileDoesNotExistException . 
 Added test directory for dumping failing build data
 Added support for uploading files with the `UploadFileVec` class
 Added logging statement to track number of idle channels closed . 
 Added new method to handle multiple initialization parameters in FilterDef and ServletDef classes . 
 Added import for java . util . concurrent . Callable and updated annotations to make CacheStats compatible with Java 11 . 
 Fixed formatting issue in Logger . java ,  improved readability and maintainability . 
 Fixed bug where callback was not being cleared after cancellation . 
 Changed how preview URL is generated for unpublished posts . 
 Fixed exception message for improved error handling
Improved handling of WordPress links in WPWebViewActivity . 
 Fixed icon for tethering notification
 Fixed case NOT_EXISTS_PARTLY_MERGED in ToBeMergedDialog . java
 Fixed issue with disconnecting subscriber in RedisBroadcaster . 
 Added support for detecting Kotlin annotations in compiled Java code . 
 Fixes layout calculation for docked tasks
 Fixed null pointer exception in AbstractDynamicMethods . java by adding missing import statements for Javautil and Apache Commons Lang . 
 Added setChunkedTransfer (  )  method for gzip encoding
Fixed a typo in the `TimeRange` import and added the `cacheTime` field to the `DashboardWidget` persisted fields . 
 Fixed bug in drag edge detection
 Changed HTTPMethod enum to include PATCH method . 
 Fixed GetConnectionTimeoutException handling in DruidDataSource . 
 Added instrumentation for InterceptionInstrumenter . java :  updated parameter handling for instrumented methods . 
 Added API council approval hide flag for ListView . 
 Added import for JRubyUtilLibrary and updated hook implementation to pass scenario result to Ruby method . 
 Fixed limit check for header size to prevent buffer expansion
 Fixed bug for Issue 603 by updating import statements . 
Fixed authorization configuration for token endpoint  (  # 73fa3fda )  by adding fully authenticated authorization for OPTIONS requests  (  # 5d5c13ee )  . 
 Added input type for password entry field to match the latest Android security guidelines . 
 Added implementation for JpaBaseConfiguration in HibernateJpaAutoConfiguration
 Fixed bug in Test . java by adding a new line character  (  \ n )  at the end of the getSingleTestProperty (  )  method . 
 Added support for initializing Git repositories in GitInit . java
 Fixed filter for source files to exclude heron / storm directory due to license issues . 
 Added method call check for Groovy language
 Added new aspect class to instrument Tomcat requests
 Fixed bug in model loader implementation . 
 Refactored data source descriptor to improve performance and reduce memory usage . 
 Fixed deprecated method in MockWebServer . java
 Fixed issue with refresh of virtual files in InspectionApplication . 
 Added support for batching multiple meshes for a single material in BatchNode  (  # 303de7a02 )  . 
 Fixes issue with sending messages in sync mode by updating sendInSync timer
 Fixed issue with text selection in WebChromeClient onSelectionStart (  )  method . 
 Fixed empty block check for compatibility with Checkstyle 8 . 0
 Fixed compatibility issue with Android Studio 2019 . 2
 Added logic to suppress turn prompts and calculate routes when necessary . 
 Added optimization to reduce size of TopNOperator . 
 Added check for connected devices before returning from handleMessage (  ) 
 Added check for resolvedUserId in getUserStateLocked (  )  for improved security . 
 Added application context to DrawerMenuAdapter
 Added logging trace for contact not found in Yahoo ServerStoredContactListYahooImpl
 Added implementation of CharSequence interface for ShadowSpannableStringBuilder
 Added skipIfStartingOnBoundary check to BinaryBlockReader to improve performance
 Added functionality to toggle fade scroll bars and scroll bars on top for all scroll panes in the test . 
 Fixed issue with missing post ID in ReaderPostListFragment . 
 Added method to construct ValidationException with a field and message
 Fixed version check in SpringSecurityCoreVersion to match current version . 
 Fixed issue with tracking reader article comment in Analytics . 
 Added remotable view method for setEnabled (  )  in TextView . 
 Commit message :  Added FontAwesome icon to ImageView in PlaygroundActivity . 
 Fixed leading zero for addLeadingZero method
 Fixed null Principal check in JbossIntegrationFilter
 Added new child view at index 10
 Generated Change Log file with updated schema and data . 
 Changed the implementation of `MacMessagesImpl` to use the `invoke` method with a single argument instead of multiple arguments . 
 Added new IOBalancer implementation for TCP connections
Added support for PGP signature and encryption in PGPDataFormatTest
 Changed license text in DaoComponent . java . 
 Changed how `longLogFactoryUtil` is handled in the `SourceFormatter` class . 
 Fixed wait time for lock acquisition from 1ms to 1s in BlockLockManager . 
 Added new property to determine whether LiquiBase should run  (  # 48da529ec )  . 
Added media ID check for FeedItemMenuHandler to prevent crash when media is null
 Added cache for directory scans in TreeSnapshotter
 Added boolean  ' myBackgrounded '  to track whether the progress window is currently in the background . 
 Fixed negative edge case in compare method
 Changed how gradient colors are calculated to ensure they are never negative . 
 Fixed issue with position calculation in MediaFocusControl . 
 Added new encryption status check based on SystemProperties . 
 Added flag Intent . FLAG_ACTIVITY_CLEAR_TOP to avoid reusing an old task in SearchDialog . 
 Fixed exception handling in CommandResultCallback to properly handle non - OK commands . 
 Added support for retrieving Android resource path from system property . 
 Added method to set maximized control for SQL editor . 
 Changed Constant . java to use a new URL for the welcome message :  updated WELCOME_MESSAGE_URL from http :  /  / brave . com / brave_android_welcome . html to https :  /  / brave . com / brave_android_welcome . html . 
 Fixed encoding detection for PropertiesLoaderSupport . 
 Added vertex ID factory and edge ID factory methods for improved indexing performance . 
 Fixed issue with expected shard size calculation in DiskThresholdDeciderUnitTests
 Commit message :  Fixes issue with incorrect property configuration in JolokiaAutoConfigurationTests . 
 Added listener for complete event to StreamSourceFrameChannel
 Fixed bug in enterTemp (  )  method
 Changed the value of the `value` field for the `TYPE` feature from `key  =  =  null  ?  null  :  key . getClass (  ) ` to `null  :  key . getClass (  ) ` in the `MapELResolver` class . 
 Added check for null duration and updated return type to avoid  - 1L
 Added method to calculate endpoint scores using penalties and severity
 Added support for new encoding encoders . 
 Added initialize method for service bootstrap initialization
 Changed the behavior of the `BoolQueryParser` to always return a query with boost factor applied ,  regardless of the number of clauses . 
 Added getBiteChance (  )  and setBiteChance ( double )  methods to Fish interface
 Added animation trigger for dim animation when launching from app with thumbnail
 Changed URL in `TaskTest . java` from https :  /  / www . url . com to https :  /  / ww . url . com and added a new URL http :  /  / www . urls . com . 
 Added exclude for UserManager instance field in AndroidExcludedRefs for SDK_INT  >  =  JELLY_BEAN  &  &  SDK_INT  <  =  LOLLIPOP_MR1 ,  and added a new exclude for DevicePolicyManager ' s settings observer for MOTOROLA devices with SDK_INT  =  =  KITKAT . 
 Fixed bug  # 12345678 :  Menu button removal issue . 
 ReaderTagTable :  added new method to retrieve custom list tags
 Fixed bug in outputBuffer . checkConverter (  )  method
 Added import for org . slf4j . Logger and updated logger configuration in ProxyResponseHandler . 
 Fixed null pointer exception when processing XML attributes . 
 Fixed null pointer exception in AmazonIapManager . get (  ) 
 Fixed null value handling in InsertGenerator to properly quote null values for SQL injection protection . 
 Added printing of throwable exception to UrlNetworkManager constructor
 Added dispose method for FocusTrackback class
 Added check for calling isUpToDate (  )  from Event Dispatch Thread to prevent deadlocks
 Changed method signature to throw a different exception type . 
 Fixed bug in handling of content length and charset for NettyAsyncHttpProvider . 
 Added intent filter for custom event handling . 
 Added new method to dismiss color fade when screen is turned off
 Fixed todo comment in MongoFind . java
 Commit message :  Added new unnamed scope and updated loadState method to include it . 
 Added break statement in DownloadObserver . java to handle download status updates
 Added property  ' coref . mentionDetectionModel '  to StatisticalCorefProperties . 
 Added debugging log statements for PatternUnlockScreen . 
 Added generic type parameter to PathRepresentation for improved type safety
Fixed issue with incorrect JRE path in Alluxio Framework configuration . 
 Added syslog - redirector entrypoint to capture std out / err in container config
 Fixed builder class name in DefaultStrategy to match the package name
 Added new index for texture scaling in TerrainEditorTopComponent
 Added idle command for AceEditorIdleMonitor :  type ,  mouseX ,  mouseY ,  modifiers . 
 Fixed issue with tag closing in JavaDoc documentation
 Fixed issue with maximum SDK version and renamed package in AndrolibResources . 
 Added new StatementLocksFactorySelector to EnterpriseEdgeEditionModule for better statement lock management . 
 Changed the implementation of `dpToPixel` to use `TypedValue . applyDimension (  ) ` instead of multiplying by the density dpi . 
 Added new moon rise and set calculations for improved accuracy
 Fixed bug in RingBuffer creation
 Added file lock disable and mmap enable if supported for crash testing
Renamed parameter name in `sendAnyPendingSMs` method in `SubscribedSession` class . 
 Added print statement for installed plugins
 Added support for setting active profile owner in DevicePolicyManager . 
 Added fallback to no encryption for highest encryption level . 
 Fixed bug in Daemon . java to prevent IllegalStateException when starting or stopping daemon . 
 Added enableHelpKey (  )  method for OutputPanel to prevent NPE  (  # 22 )  . 
 Added constructor for ViewDelegate with newInstance method signature
 Fixed issue with network state machine ,  updated sendNetworkStateChangeBroadcast (  )  to include mLastBssid . 
 Added implementation for JetPackageDirective to implement JetExpression . 
 Added default value for  ' network_threads '  to avoid configuration error . 
 Changed how the author ' s name and email are displayed in the DetailsPanel to include the commit time if different from the author time . 
 Changed method `queryComment (  ) ` to only replace comment if not disabled . 
 Added magic value to mark RecordsTable as dirty . 
 Added hasStableIds to DrawerAdapter for improved performance
 Fixed exception type for long parameter parsing
 Added check for editor mode in VimShortcutKeyAction to prevent accidental insert mode activation . 
 Added getSelectedColorName (  )  method to return selected color name instead of null
 Added support for null return values and improved media type handling in AbstractMessageConverterMethodProcessor . 
 Added test methods for GroupProperties and GroupProperty classes
 Fixed bug in Jingle Nodes discovery logic for account   +  accountID . getAccountAddress (  )  . 
 Added null check for getStringValue (  )  to prevent NPEs
 Added isActionDown (  )  ,  isActionUp (  )  ,  isActionMove (  )  ,  isActionCancel (  )  ,  and isActionOutside (  )  methods to TouchEvent class
 Added ability to specify credentials when performing requests
 Added support for detecting player direction in WorldEditController
 Changed how the `TaskViewServiceTest` class handles the `viewModels` list to include additional assertions for the types of the models . 
 Added new option to select preset file patterns in Find In Files Dialog
 Commit message :  Changed InfinispanCacheStoreManager to support distributed mode . 
 Added empty state layout for ListView in ListActivity
 Fixed bug in LocalBlockInStreamIntegrationTest related to Tachyon cluster resources . 
 Added method `putJSONAA4 (  ) ` to AutoBuffer class
 Fixed buffer size for write operation in WebSockets class
 Added new purpose serialization for ASSURANCE_CONTRACT_CLAIM ,  ASSURANCE_CONTRACT_PLEDGE ,  and ASSURANCE_CONTRACT_STUB . 
 Commit message :  Changed LdapResourceCRLFetcher to LdaptiveResourceCRLFetcher in AbstractX509LdapTests . 
ReverseProxyHandler updated to include ExclusivityChecker for upgrade requests . 
 Fixed bug in saving VM options for OutOfMemoryDialog . 
 Changed code to ensure `JvmThreads` is greater than or equal to 0 . 
 Fixed issue with converting number schema with non - integer values . 
 Update to cache layout version 14 . 
 Added Languagetool as a top suggestion for words containing  ' Languagetool '  or  ' languagetool '  . 
 Improved Find Dialog functionality with added action listeners for scope controls and find button
 Added implementation for AppCallbackHandler class
Fixed issue with incorrect execution state checks in MulticastManager . 
 Added import for com . apple . eawt . AppEvent in ArgDelegationActivator . java
 Added support for generating worlds with mixin world types
 Added validation for dataset format in Parse . java
 Added custom name rendering for GotoRelatedItem in GotoRelatedFileAction
 Added new shard rewriter for OrcStorageManager
 Added check for timeout expiration in LockPatternUtils . 
 Added invalidate (  )  method to GhostView to update display list
 Fixed issue with setOpenOnTapEnabled method
Reduced method signature length in `ExpressionReplacingVisitorSupport` . 
 Fixed hash code calculation for AntPathRequestMatcher . 
 Update DisplayPowerController to handle screen state transitions for doze mode . 
 Added `setVisible ( true ) ` instead of `show (  ) ` to improve performance
 Added entity types to World class
 Added check for public method of non - public class and updated method invocation to reflect hierarchy . 
 Fixed bug in testPostWithContinuation (  )  causing incorrect response . 
 Added new line of code to improve efficiency in `getSuggestionItemHeight (  ) ` method . 
 Added method to get private IP addresses instead of private DNS names in AWSClient . 
 Added import for IElementType and updated constructor for HtmlDocumentImpl
 Changed  + import android . util . Config ;  to  + import android . util . Config ;  in ViewDebug . java . 
 Incremental changes to InMemorySessionManager and SessionManager
Added test class replacements for Paint and Canvas
 Changed execution timeout from 3 to 4 milliseconds in HystrixCommandTimeoutConcurrencyTesting . java . 
 Changed implementation of `WXScrollView` to calculate height of sticky view correctly . 
 Added support for OkHttp version detection in User - Agent header
 Improved Builder pattern implementation for Hero objects ,  now with simplified constructor and reduced exponential constructor growth . 
 Added variable descriptor to frame map
 Added new ownership information to the NodeCmd class . 
 Changed how `deadReceivers` is initialized and added a new listener . 
Fixed log statements for improved debug output  (  # 359 )  . 
 Added logging for Travis CI server to only run agent application with a single type locator when running out of memory . 
 Added HTTPS scheme to Flickr API endpoints for improved security . 
 Added support for handling  @ Provides methods on classes with non - public visibility and abstract classes . 
Added import for com . intellij . util . containers . ContainerUtil in ChangeListWorker . java
 Added implicit lock interval field to IndexerCoordinatorResource
 Fixed issue with null pointer exception when selecting remote repository in CurrentBranchDialog . 
 Changed handling of connection persistence in EarlyCloseServlet to prevent potential issues with non - persistent connections . 
 Fixed bug in handling of large intermediate results for GroupByQueryEngine . 
 Fixed issue with JavaScript timers not resuming correctly in background . 
 Update PhoneStatusBarPolicy to display  ' Quick Settings '  Bluetooth toggle button instead of  ' Bluetooth Disconnected '  accessibility label . 
 Added new connection handling method in JedisSlotBasedConnectionHandler
 Added check for empty  ' resource '  element in IQBindHandler . 
 Fixed scrollbar visibility logic to match new API . 
 Fixed ClassLoader usage in ClassResolver to prevent class initialization . 
 Fixed formatting issue in EvictionHelper . java
 Added method to continue running completion in combo editor . 
 Added test for rejected execution exception in ClientExecutorServiceTest
 Fixed issue with remote metadata update in VoldemortAdminTool . java ,  added missing system store to map . 
 Added vertical scrollbar to UIDropdown widget
 Added delete extractor method for input with ID  <   +  inputId  +   >  . 
 Commit message :  Changed listener . onEntriesChanged (  )  to be non - modal . 
 Update to use MD file for GCM key instead of TXT file . 
 Optimized class ATNOptimizer ,  improved performance for grammar optimization . 
 Added check for disposal before updating search results in LivePreviewController
 Added import for SpongeAPI ' s Player class
 Added new network factory registration logic for improved performance . 
 Fixed issue with progress bar not updating correctly
 Added sub command execute (  )  method in PinotAdministrator class
 Added mapping for  ' bmp '  to  ' image / bmp '  in MimeMappings . java
 Added new method to create TagDefinition instances in MockTagDefinitionDao
 Added toString (  )  method to SkylarkNestedSet class
 Fixed bug in TypeConverter for ParseException
Added CONTINUE_WAIT and THROW_EXCEPTION options for ExceptionAction ,  and updated Invocation to handle them . 
 Added item ID to fill method in BlockTank . java
 Added 20 %  width to indent view in DraggableTaskListActivity
 Changed command line options for debugging and Java debugging . 
 Fixed debugging log statement to improve readability . 
 Added license information for Apache License ,  Version 2 . 0 . 
 Added wrapper for collection manager class in ODistributedStorage
 Fixed bug in password hashing algorithm in ServerRestInterfaceRealm . java
 Added PyOverrideTest class
 Update DeleteMountPointEntry to match JournalEntryType . DELETE_MOUNTPOINT
 Added ability to print stack trace to console view
 Vector3 :  Changed behavior for division by zero
 Fixed bug in HibernateSession . java by updating import statements for improved code compatibility . 
 Changed the `LocalTachyonCluster` field from `private` to `final` in the `TachyonFSContract` class . 
 Added new activity exemptions in PasscodeManager . 
 Changed text color of action button in Snackbar . 
 Added new features for distSim in NERFeatureFactory
 Changed how file watcher is started in ProjectManager . 
 Fixed issue with loading portal . properties file in SourceFormatter class . 
 Changed `HashSet < String > ` to `LinkedHashSet < String > ` in `commandParams` . 
 Added support for transitioning a select statement to a table
 Fixed near cache config test method to use the correct method signature
 Changed PyElementType DECORATOR_LIST to PyDecoratorListElementType . 
 Fixes NinePatchDrawable constructor to use new Rect (  )  instead of null for better compatibility
 Added ProgressManager . checkCanceled (  )  to ensure cancellation of progress during fragment creation . 
 Added fileChannel (  )  method to Image class for accessing raw log
 Fixed bug in BaseInspectionVisitor to prevent duplicate messages for nested and anonymous classes . 
 Fixed constructor for HttpGet with URI parameter
 Added method parameter for firstElement and secondElement in Node class
 Fixed bug in ProviderTestCase2 . java by updating the mock context class
 Added new route calculation logic for forward direction and reset evalWaitInterval to 3001 for new routes . 
 Added empty constructor for ClassFilterEditor
 Fixed exception handling for XMPP reconnection attempt . 
 Added incomingEdges field to DefaultImmutableDirectedAcyclicGraph
 Changed the implementation of getDragView to handle edge cases for collapsed items . 
 French localization of AbstractDateCheckFilter
 Added method  ' request ( RequestMatcher matcher )  '  to BaseServer
 Upgraded PreStartupStoreUpgraderTest . java with changes from b4e2721 to 4c6f2f6
 Changed implementation of `WebApplicationHeader` to display user identity in the header . 
 Added for loop to create all children PSI in CompositeElement . 
 Added forceChangingFlag method to update isChanging field in DependencyDescriptor . 
 Added filter query for message count in MessageGateway
 Fixed issue with empty node list in TransportClientNodesService . 
 Update PhoneStatusBar to use new KeyguardViewManager implementation
 Changed how `MediaRecorder` profiles are set based on camera ID . 
 Added bindings for DefaultLocalClientChannelFactory and DefaultLocalServerChannelFactory in NettyModule . 
 Changed implementation to suppress internal errors caused by bugs in JDI implementations . 
 Added support for Consul API endpoint syntax :  `consul : apiEndpoint` . 
 Fixed bug  # 12345678 in NioServerSocketPipelineSink to avoid infinite loop . 
 Added support for tracking blocks of interest in formatting . 
 Commit message :  Fixes issue with KeyguardManager proxy in AudioService . java . 
 Added implementation of MultiEnumeration for ReplicatedContext to improve performance
 Fixed null pointer exception in GlobalAuthenticationPolicyWebflowEventResolver by returning null when service or authentication is null . 
 Fixed bug in UserIncludeMethodInterceptor to properly handle trace recording for user traces
 Fixed floating point bits conversion to ensure correct sorting . 
 Added catch block for InternalException in DebugProcessImpl . 
 Added ttl field to CassandraDatastore class  ( afc8efd .  . facfc2e )  . 
 Added fuzzy search query for  ' Name : Builder '  to index  ' bobTheIndex '  . 
 Added shutdown hook thread with name  ' hz . ShutdownThread '  . 
 Added method to setExtendedState for improved frame visibility
 Suppressed warnings in ErrorHandlingTest to test scope handling
 Added support for max - size config on map in LocalRegionCache . 
 Added documentation and analysis for the  ' answer '  method in ClonesArguments . 
 Fixed deprecated methods in JspContext and JspException to match JSP 2 . 1 specification . 
 Added new SDK  ' PyDetectedSdk '  and updated SdkListModel in PyActiveSdkConfigurable . 
 Added throw statement to methods to indicate unsupported functionality . 
 Fixed bug causing incorrect highlighting of duplicate symbols in editor
 Fixed issue where main controller ' s switchToBubbleView (  )  was not being called when there were multiple tabs open . 
 Fixed formatting issue in ConfusionRuleEvaluator . java for better readability . 
 Fixed issue with incorrect file name in TxLog . 
 Added API response element for error in trip planning . 
 Added TEXT_INSET_H to 20 for improved layout . 
 Fixed issue with OpenComputers handling player interaction events . 
 Fixed deleteFromRelationshipIndexnoValue method to use  @ PathParam (   \ key \   )  instead of  @ PathParam (   \ id \   )  for better compatibility . 
 Fixed zoom animation issue in MapVectorLayer . 
 Changed behavior of `DistributedStoreManagerTest` to log range information differently . 
 Fixed formatting issue in ignored_cols log message
 Fixed camera prewarm service class name in KeyguardBottomAreaView . 
 Added WildcardConfigPatternMatcher for InstancePermission
 Fixed bug in multi - import resolution by considering entire names ,  not just the first component  (  # 9453 ) 
 Added setVisible ( boolean )  method with animation cancellation
 Added support for BroadcasterCache in XHRTransport
 Added default tick tuple interval for HiveBolt based on topology timeout
 Fixed bug in CasAuthenticationToken . java by removing unnecessary print statement
 Added org . jetbrains . annotations . NotNull annotation to method signatures in AppEngineEnhancerBuilder class
 Added FML marker for server modding
 Added null tag to tabToAddTo in BottomBarBadge . 
 Changed the comparison order in the `TreeWalker . java` file to improve code quality according to Checkstyle rules . 
 Changed implementation of `Tokenizer` class to remove unnecessary comment . 
 Added default value for empty iterable in Iterables . java
 Added public method for getting archive handler in VfsImplUtil
 Fixed internal error in SpiderAPI . java for proper handling of URL not found . 
 Changed file SarSubsystemTestCase . java to include a new method getSubsystemXsdPath (  )  and updated the getSubsystemTemplatePaths (  )  method to include a new template file . 
 Fixes bug causing empty sliding window for tenured pool . 
 Added new method to recurse through directions in EditSession . 
 Added import for Paths class
 Added comparison method for EdgeEntry objects
 Added extra data for USE_LOCATION_FOR_SERVICES in Browser . java
 Fixed issue with icon switching when dot org icon is selected . 
 Improved worst - case time complexity from O ( n^2 )  to O ( n * log n )  for quicksort . 
 Added toCharArrayForSpringWebSocket method for Spring WebSocket support
 Added method reduce (  )  to FrameUtils class
 Fixed bug  # 1234 :  Rendering issue with null path in TextEditingTarget . 
 Changed package import statements to include additional packages . 
 Fixed bug in JarClasspathSnapshotFactory to avoid unnecessary loading of snapshots . 
 Added  ' urls '  field to ContactAccessorSdk5 class to match API changes . 
 Added dependency on JUnit 4 . 11 to GrailsCoreDependencies
 Fixed bug in LineMarkerProvider implementation for Python
 Added constraint undo logic for index removal
 Added method `timeoutOccurred (  ) ` to handle connection timeouts in Peer class
 Added SensorManager binding to RoboGuiceModule
 Fixed issue with slow servers in XmlJsonStressTest . 
 Added testIgnored method to ignore issue  # 9637
 Fixed issue with detached views not being properly cleaned up in RecyclerView . 
 Added id field to Node class
 Added new GBM test case for multinomial classification with alphabet categorical feature . 
 Fixed bug in parsing JSON array of child views
 Changed method signature to use String . format (  )  for building the configuration name . 
 Changed class name from Study project to Learn Python . 
 Changed the `MEMBER_LIST_PUBLISH_INTERVAL_SECONDS` property from `60` to `0` . 
 Added Stage . setKeyboardFocus (  )  method to Window for better keyboard handling . 
 Fixed issue with forceEchoOnExec (  )  in TextEditingTarget . java ,  now correctly set to true for knitr and false for pdf compilation . 
 Added support for HiDPI displays
 Fixed method signature to match updated interface
 Fixed exception handling in TransferOnePidgeyExample class . 
 Added check for null account / file before checking for upload in FileDownloader
 Fixed return type of getter method to avoid NPE
 Added custom POI filter functionality to MapActivityLayers
 Fixed exception message for lock acquiring in TruncateTranslogIT test . 
 Fixed method signature for isOk (  )  to match the updated response format . 
 Fixed issue with suppressed messages in MessagingController . 
 Refactored implementation of IntroduceVariableBase to handle outermost parenthesis and PsiNewExpression . 
 Commit message :  Improved query performance by optimizing component handling . 
 Fixed issue with non - ASCII characters in Processing installation path . 
 Added progress bar visibility handling for Notification . java
 Fixed naming inconsistencies and removed unnecessary lines in ViewportTest1 . java
 Fixed GrailsUtil . getGrailsVersion (  )  to return correct version . 
 Fixed issue with account number calculation in MessageProvider
 Fixed issue with map interceptor instance aware test . 
 Added regular expression pattern matching for highlighting output in PyConsoleHighlightingUtil  ( 6367aa5 )  . 
 Fixed file content in MavenTestCase . java to match expected value . 
 Added default cluster name ,  index key ,  and remote host to ElasticsearchClient class
 Added premultipliedAlpha property to ParticleEmitter class
 Fixed issue with negative indices in IntStream . java
 Added method `StreamComparator (  ) ` to `SpdySession` class
 Changed code added to move blocks among tiers . 
 Fixed interface check for Compiler . java
 Added `e . getStateChange (  )   =  =  ItemEvent . SELECTED` check to update receiver parameter in JetChangeSignatureDialog
 Added log . debug statement to print principal id for service . 
 Added support for NIO data input stream in IncomingTcpConnection class
 Changed how exceptions are caught and logged in the `AssetExtractor` class . 
Added method `asByteBuffer (  ) ` to ByteString class
 Added support for new query queue definitions . 
 Changed behavior of managed transaction factory test to pass . 
 Added new expected type  ' com . linkedin . restli . examples . greetings . client . finders '  to TestResourceSchemaCollection . 
 Fixed boolean value for SYSTEM_LOG_TRANSACTIONS config option . 
 Fixed SQL literal escaping for BIGINT ,  BIT ,  BOOLEAN ,  DECIMAL ,  DOUBLE ,  FLOAT ,  INTEGER ,  NULL ,  NUMERIC ,  SMALLINT ,  and TINYINT types . 
 Added override for equals method to compare EstimatedHistogram objects . 
 Added method to check if Java version matches given vendor
 Fixed bug in ProxyInfo creation after method execution
 Added method getDimensionServiceKey (  )  to GitFetchDialog . 
 Added skip null reference check for arrays in StringCollector . 
 Fixed ActorPush endpoint registration in ActorSDK . java
 Fixed bean definition for MessageBrokerBeanDefinitionParser to include SockJS task scheduler . 
 Fixed exception handling in PinotFileUpload class . 
Fixed a bug in NavigatePointFragment by updating the location variable correctly when the user navigates to a new location . 
 Added JNI method for printing debug messages
 Fixed bug in MessageDigest initialization
 Improved error handling for timed - out timer threads . 
 Fixed bug in HiveSessionImpl to correctly handle empty opHandleSet when calling release (  )   (  # 1154 ) 
 Fixed edge . field (  )  method to return correct label field value . 
 Fixed Nullable annotation on method to make it non - nullable
 Added import for resolved libraries in AbstractExternalProjectImportBuilder
 Added openSocket field to Http11AprProcessor for improved performance
 Changed from preserving windows to not preserving windows in `resizeStackLocked (  ) ` method . 
 Added new SourceHttpMessageConverter for XML support in RestTemplate
 Added check for deprecated warnings in whitelist . 
 Fixed buggy readAvailable (  )  implementation to prevent infinite loop in non - blocking mode . 
 Fixed error message for null class in JavaDocInfoGenerator
 Added ability to set session timeout in UsersResource
 Changed behavior of onActivityResult (  )  to handle RESULT_CANCELED case . 
 Added stopActivityMonitor (  )  method to ActivityUtils for proper cleanup
 Fixed nested thread group issue in WebappClassLoader
 Added dependency on yaml package version 2 . 1 . 5 . 
 Fixed type check for WebSocketProtocolStream to improve compatibility . 
 Fixed exception handling in FlashlightController to prevent crashing . 
 Added method `getBoolean ( String ,  boolean ) ` to BasicBSONObject . 
Update App . java and AbstractApp . java to return the server this app runs in ,  and add a new method to GraphDatabaseApp to return the GraphDatabaseShellServer . 
 Fixed issue with mods attempting to allocate chunkloading tickets beyond their maximum limit . 
 Added new GraphService implementation with improved performance . 
 Added weak hash map for already applied AST nodes to improve performance . 
 Added setting for concurrent streams in recovery . 
 Fixed ignored test for manual proxy setup
 Fixed issue with ContentProvider shutdown in ProviderTestCase2 . 
 Added  ' isRootNamespace '  parameter to ContentViewLocal and BigContentViewLocal for better performance
 Added new theme tags :  holiday ,  photoblogging ,  seasonal
 Fixed layout injection for NavigationBarInflaterView . 
 Added logic to skip GC warning listener for concurrent collectors . 
 Fixed type check for unsubscribe presence subscription
 Added player . getNetworkSynchronizer (  )  . setPositionDirty (  )  to update position in network synchronization . 
 Fixed layout configuration for MagicLinkSignInFragment . 
 Fixed bug causing incorrect configurables registration for removed options
 Added setConfig (  )  method to MortarDemoActivity to fix instance configuration issue
 Added exclude for InputMethodManager :  : servedView when SDK_INT  >  =  ICE_CREAM_SANDWICH_MR1  &  &  SDK_INT  <  =  M . 
 Added support for double weights in ZParams
 Added getExecuteTimeMillis method to Timer class
 Fixed issue with origin repository name comparison in RepositoryManager . 
 Added REAL_LANGUAGES array to exclude demo language
Added support for plugins to handle link clicks and intercept URL loading in CordovaPlugin . java ,  and modified PluginManager to return false to allow URL loading by default . 
 Added support for listing X attributes on Mac with the `listXAttr (  ) ` method . 
 Added import for DumbAwareAction . 
 Added support for activation function in RBM layer
 Added support for Java 11 module types
 Added new values to the  ' place_of_worship '  and  ' internet_access '  fields in the PoiFiltersHelper class ,  and updated the  ' OTHER '  field to include new values for  ' internet_access_wlan '  ,   ' internet_access_wired '  ,  and  ' internet_access_terminal '  . 
 Added method  ' ls ( String ,  boolean )  '  to return a list of files / directories under a given path . 
 Changed how IOHelper handles unspported charset exceptions . 
 Added loadLibrary (  )  call for RSSupport and rsjni libraries
FilesPane :  Update FilesList to display file paths correctly . 
 Added support for setting the size of the GIF file in the AnimatedGifEncoder class . 
 Fixed bug causing incorrect account order in Preferences . java
 Added isDumbAware method to CommonActionsPanel for proper update behavior when editing a JTable . 
 Fixed hack to extract jnilib to dylib on OSX Java 1 . 7 + 
 Fixed issue with ScrollView interception during swipe gestures  (  # 1078 ) 
 Fixed issue with task list metadata initialization . 
 Changed how assertThat (  )  calls are formatted to be more consistent . 
 Lazy initialization of HttpsConnection ' s SSLSocketFactory moved to static block for improved performance
 Fixed bug causing incorrect handling of Throwable in ServletException constructor
 Added EnglishWordRepeatRule and AvsAnRule classes ,  and updated WhitespaceRule and LongSentenceRule classes . 
 Fixed bug where unrecycled bitmaps were not being properly recycled . 
 Fixed bug in GitChangesCollector to properly handle files under previous files . 
 Added method to retrieve GPUImage instance used by view . 
 Added ChineseGrammaticalStructure support
 Fixed connect timeout issue in NettyAsyncHttpProvider
 Fixed issue with hidden objects in SQL completion suggestions
 Added method to retrieve Github username and repository name in AbstractGithubTagDownloadedProjectGenerator
Fixed issue with file processor availability in UriReaderTest
 Changed the size of the `array` field from 14 to 10 . 
 Added import for ActionBarSherlock library in TraktSyncActivity . 
 Fixed ignore annotation in TestStoreCopy . java to make build green . 
Added timestamp provider to ConsistentKeyLocker to use microsecond precision  (  # 503 ) 
 Fixed bug in getBaseDir (  )  method to correctly retrieve current directory . 
 Added precisionThreshold field to CardinalityAggregatorFactory
 Added visibility check for thumbnail strip in ReaderPostAdapter . 
 Fixed issue with importing JSON data into Refine
 Fixed issue with excessive background exceptions in ConnectionState class  (  # 351 )  . 
 Update to correct indentation and line breaks in StandardService . java
 Changed method `setOrigin` and `setDestination` to use `makeGenericLocation` instead of `routingRequest . from` and `routingRequest . to` . 
 Fixed bug in NetworkAvailabilityChecker implementation to properly handle empty socket addresses . 
 Commit message :  Changed test deployment to use new impl - level test cases . 
 Added check for missing arguments in rule definition . 
 Commit message :  Fixes issue with Jackson 2 . x XML converter tests . 
 Added testCoGroupKeyMixing2 (  )  method with updated code for CoGroupOperatorTest . 
 Added support for edge type label class in DefaultProcessor
 Added checkCallingPermission for startLockTaskModeOnCurrent
 Added ability to specify authors for Git checkin . 
 Added augmentation for accessibility events in PhoneStatusBarView to include content of status bar . 
 Added check for natural sorted set of methods before iterating over annotations . 
 Fixed error handling in TextView . java
 Fixed bug in ConfigFileApplicationListener to properly handle null or non - existent resources . 
 Fixed bug in clone method for Spatial class
 Added check for visible behind activity when moving home task back to maintain user visibility . 
 Changed BuckConfig to include additional ignore fields for daemon restart . 
 Changed how `nextRequest` is called to pass a new `Request` object and `Future` instead of the original ones . 
 Added new dependency to FrameworkSupportModelBase class
Updated ContactListPanel to handle updating chat contact status with new parameter protoContact  ( instead of MetaContact )  for improved performance and flexibility . 
 Fixed formatting issue in TreasuryYieldReducer . java
 Added explicit naming requirement for IsNamedMatcher
 Copied SquidDatabase files to folder :    +  toFolder
 Added import statement for java . util . Arrays
 Fixes issue with rounded corners in ReactImageView by updating scaling transform
 Fixed issue with metadata serialization in LeveledManifest . 
 Added new RuntimeException message for shutdown (  )  method
 Commit message :  Added new tab listener to FeatureToggles class . 
 Added import for org . elasticsearch . AbstractNodesTests in NoMasterNodeTests . 
Update ZWaveNodeStageAdvancer to handle WAIT state more efficiently
 Changed method `testMultipolygon` to exclude disabled test due to discussion with author  ( AMB )  . 
 Fixed handling of interface down events for Tethering service
 Fixed bug in HttpClient 4 . 0 . x to prevent UnknownHostException on occasion . 
 Fixed ViewConfiguration . getTapTimeout (  )  in SimpleTwoFingerTapDetector for better accuracy
 Fixed issue with mouse event handling in InplaceEditor . 
 Changed method `waitFor (  ) ` to return `exitValue (  ) ` and added an assertion to check the exit value . 
 Added support for multiple users in UserManager . java
 Fixed JDBC 3 . 0 getParameterType call not supported in StatementCreatorUtils
 Added permissions for JndiPermission class
 Fixed method signature in HelloBean to match updated interface
 Added check for null internalRoot in OpenExternalDirectoryActivity . 
 Fixed XMLRPC API import to use Apache ' s WebServer instead of IdeaAwareWebServer
 Added nullable `after` parameter to `VcsLogDateFilter` constructor for improved flexibility
 Changed implementation of inner classes to ignore abstract inner classes . 
 Changed code to compress wildcard type in `JavaWriterTest` test method . 
 Added method visitor for FrameAdvice in AdviceTest . 
 Fixed pattern validation for allowed content types in BinaryHttpResponseHandler
 Fixed bug causing tiles to flip incorrectly in vertical direction . 
 Added new config property  ' numStandbyReplicas '  to control the number of standby replicas for each task . 
 Fixed dependency collection to match initial values . 
 Added Comparator for modifier duration in descending order
 Fixed copyright year and updated license terms . 
 Fixed bug in fluid block registration . 
 Fixed formatting issue in ActiveThreadCountService . java
 Added body (  )  method to convert inbound message body expression to expected type
 Added method to return QueryFilter object for slice criteria
 Fixed null pointer exception in PeopleManagementActivity . java by adding a check for empty string in getDotComBlogId (  )  method . 
 Changed import statement from `ODatabaseDocumentTx` to `ODatabaseDocumentTx` in `DbImportTest . java` . 
 Added flag FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY to Intent for launching activities from recents view . 
 Fixed bug in read method to properly handle buffer size
 Changed how commit message is substituted to preserve previous message . 
 Added optional `directed` field to `NetworkBuilder` for better type safety
 Added new handler to process presence updates and updated map to include new handler . 
 Changed method signature for `showUpwardsFoldToken (  ) ` from `PrefValue < Boolean > ` to `PrefValue < String > ` . 
 Added new method `setAspect (  ) ` and updated `onViewAdded (  ) ` to handle new view correctly
 Added builder class for creating headers with multiple values . 
 Added copyright notice and updated licensing information
 Fixed issue with disposing of tabbed content in VcsLogTabsWatcher . 
 Added TODOS for thread - safety in DependencyResolutionLogger
 Added separate subscription for contact status in JsBindingModule
 Fixed issue with fluid stack matching in BptBuilderBlueprint . 
 Upgraded nodes in test cluster . 
 Fixed compatibility issue in QueryPhase class by removing unused  ' filter '  field . 
 Improved support for specifying file size limits in human - readable formats . 
 Added API binary directory to JvmComponentPlugin
Fixed bug in pipe events interface to make it compatible with Java 11 . 
 Changed how the `OsmandMapTileView` handles touch events to improve performance and user experience . 
 Added version number to MulticastZenPing response
 Added method to set instrumentation template file and updated recordFunctionInformation field . 
 Added new preference for default application mode . 
 Fixed method signature in BoxConsumer to match API changes . 
 Fixed method signature to match updated Java syntax . 
 Added new parameter numberOfSlots to DefaultMemoryManager constructor
Fixed issue with unsupported statement in QueryVisitor . java by removing unused code and updating to only handle CursorNode . 
 Fixed testBasic method to use Jps
 Added method body ( byte [  ]  bytes )  and body ( InputStream body )  to AtmosphereRequest class
 Added flashlight tile implementation
 Added API_DOMAIN constant for API URL
 Fixed the maxSleepTimeMs calculation to prevent misconfiguration . 
 Fixed type evaluation for null class loader and updated virtual machine comparison
 Added import statement for FileNotFoundException and updated recovery status code to handle exceptions when deleting files . 
 Changed how type information is obtained for column values in OrcTester . java from getting it from the type name to using the `getTypeInfoFromTypeString (  ) ` method . 
 Added alias for class   +  cls . getShortName (  )   +   in OConsoleDatabaseApp . 
 EncryptedPrivateKey updated :  added derived AES key for password - based encryption
 Dependency updates for Packrat and RMarkdown in DependencyManager . java . 
 Changed implementation of `hasEnrolledFingerprints (  ) ` to check for correct user ID before querying for fingerprints . 
 Added DISALLOW_OUTGOING_CALLS and DISALLOW_SHARE_LOCATION permissions to AppOpsManager class . 
 Added new command  ' data - index '  to index a Wikipedia XML dump and / or a Tatoeba file . 
 Added getClusterLastEntryPosition (  )  method to retrieve the position of the last entry in the cluster . 
 Facade implementation updated to use new DummyHolderFactory . 
 Fixed issue with item equality comparison in LookupImpl . 
 Changed how the brightness is updated for decreasing commands to use `decreaseBrightness (  ) ` instead of `setBrightness (  ) ` . 
 Fixed bug where multiple createSession (  )  calls were ignored  (  # 195945d )  . 
commit message :  Updates ResultMatchers to use a more appropriate matcher for hasSingleFailureContaining . 
 Fixed compilation error for non - static method in DeepLinkProcessorTest
 Added idle timeout configuration options
 Added tab size calculation based on document property
 Fixes issue with cursor position restoration in Editor class
 Added support for creating / modifying POI types . 
 Added more information for rule matches
 Added import for ProjectCoreUtil and updated condition for resolving files in the project . 
 Added parameter validation for  ' File '  type in ClientApiMain class
 Added new line to CubeMapMaterial . java for point light calculation
 Changed how highlighting is done for deep comparisons . 
 Added new author names to BookResultInfoRetriever class
 Added exception handling for deleteBlob (  )  method in BlobStoreIndexShardGateway class . 
 Added map field to DirectedMultiGraph for improved performance
 Added query method to return result directly without needing to traverse the tree
 Added Maria DB driver to DatabaseDriver enum
 Fixed bug in PerformanceWatcher to correctly handle empty stacktraces
 Added new value to MapStore using executeOnKey (  )  method
 Fixed version check in AndroidAnnotationProcessor to handle unknown version . 
 Added support for GoRunFileConfiguration with custom file suffix
 Added threshold for lockTime :  below this value it is interpreted as block number ,  otherwise as timestamp . 
 Fixed exception handling in ConnectionManager to properly close the socket channel on failure . 
 Added postWrite (  )  method to Vec class
 Fixed issue with external payment plugin for invoice payments
 Fixed issue with selecting VCS log content in Changes view . 
 Changed how bearing is calculated to suppress turn prompt for recalculated still - opposite route . 
 Added support for reading chunks with length greater than 2^16 - 1 . 
 Changed implementation of collectReadonlyFiles to use ModuleStoreImpl instead of ModuleImpl . 
 Fixed bug in XPath loop implementation
 Fixed bug in console pane maximization logic . 
 Fixed issue  # 376489 :  added trace message for flush operation
 Added default Kryo factory class to Heron config . 
 Added method  ' withSelectedItem ( int )  '  to set the identifier of the item to select upon start ,  and updated the selection logic to use the identifier instead of the position . 
Added import statement for GroovyObjectSupport in StreamCharBuffer class
 Added sorting to the list of plugins in the PluginsAdvertiserDialog to improve user experience  ( 495ff7b .  .  . c749111 )  . 
 Fixed issue with node names in GrammaticalRelation . java by updating references to match new node names . 
 Added maximum image width for embedded images in CommentAdapter
 Fixed bug in border handling for dragEnter event
 Fixed issue with movie not being added to watchlist / collection after successful query . 
 Added compacted marker component to SSTableReader . 
 Added ability to specify icon as a resource ID in the constructor
 Added catch block to handle exceptions when deleting temporary directory . 
 Added ability to print to console with custom text attributes . 
 Fix for double tap gesture recognition by reducing the touch slope for paged scrolling from 3 to 2 dips . 
 Added max lines for title text view in ItemActivity
 Added temp file path for recording in Android 11 and above
 Fixed deprecated method signature in AbstractCamelEndpointFactoryBean
 Fixed issue with date comparison by comparing the correct types  (  # 37 ) 
 Fixed bug in GrailsConfigUtils to correctly handle null parent context . 
Fixed bug in GLRM model implementation by updating `drop_const_cols` parameter . 
 Changed `assertCurrentActivity` method to use ` @ SuppressWarnings` to suppress a warning message . 
 Added new method to collect match results in CollectingMatchResultSink class . 
 Added support for loading default broadcaster class from framework configuration
 Fixed bug in CastManager . java to handle player state updates correctly . 
 Fixed issue with default permission template for project creator on component creation . 
Added support for updating file name and rid in OServerCommandPostUploadSingleFile
Added import statement for RuntimeIOException in PagesResponseWriter to handle EOF exceptions when writing data to client . 
 Added NoOpDNSToSwitchMapping class to prevent reverse DNS lookups
 Added import for AllocatedResource class
 Fixed JavaDoc for MethodTemplate
 Added configuration option to specify UID table . 
Added new instance of VisibleGraphImpl for PermanentGraphImpl with updated parameters
 Added method to undemote a formerly demoted contact
 Renamed clear button to clearUserDefinedDisplayName in RenameContactDialog . 
 Fixed issue with incorrect indentation in SourceFormatter . 
 Added static final string logtag for TWSpannableGridLayoutManager
Fixed bug in EqualsWithDeltaTest where actual and expected values were not properly compared  (  # 345 )  . 
 Fixed nonce calculation in Peer . java
 Added check for peer certificate chain in Http11NioProcessor to handle SSL / TLS renegotiation . 
Added support for JetBrains Annotations in JUnitUnusedCodeExtension  (  # 8503016 )  . 
 Fixed deserialization of XML node data for ZWaveController class . 
 Changed the calculation of `LOG_META_DATA_LENGTH` to use the maximum length of the partition headers instead of the maximum length of the default frame headers . 
 Changed `tearDown (  ) ` to `try { tearDown (  )  ;  } catch  ( Throwable th )  { th . printStackTrace (  )  ;  }` in `IdeaTestCase` . 
 RenamePyFunctionProcessor updated to include import statement for ApplicationManager . 
 Fixed limit and offset templates in CUBRIDTemplates class . 
 Improved performance of smart reformat on big file in JavaSmartReformatPerformanceTest . java . 
 Fixed issue with artifact cache execution
 Fixed bug in visitFunctionCall (  )  to properly handle enclosing aggregate . 
 Fixed issue with boundary string in NanoHTTPD . 
 Fixed bug causing incorrect repository path or URL in MavenIndex . 
 Added constructor parameters for m_alignSampling and m_exhaustive
 Added second storage for proxy settings compatibility
 Added StreetEdge edge type in State class
 Fixed bug in filtering logic for TreeContactList . 
Added OR matching type to Stream interface
 Added getMomentum (  )  and setMomentum (  )  methods to Entity interface
 Fixed issue with dark mode icon in FavoritesTreeFragment
 Fixed layout issue in UIDropdownScroll widget
 Added lambda parameter to DecayingDelayTripTimes constructor
 Fixed bug in AbstractAndroidMojo . java by updating the type of the  ' allArtifacts '  variable from a list to a set . 
 Added getWhereCond (  )  method to retrieve the where clause as a string
 Added support for displaying HTML content in the book info activity
 Added ignoreUnknown = true to Jackson annotation for Job class
 Fixed issue where page size exceeded max download size ,  now logging error and returning status code accordingly . 
 Added method to close files before renaming in JarBuilder
 Changed the `SilhouetteCoefficient . calcSilhouetteCoefficient (  ) ` method to return correct values . 
 Added new policy for UID 1000
 Added new method to return all active apn types in GsmDataConnectionTracker . 
 Fixed class name resolution issue in Androlib . java
 Added method  ' finish (  )  '  to  ' TranscriptScreen '  to properly release resources after activity closure . 
 Added support for inline parameters in Factory class
 Fixed issue with parsing boolean values
 Added  ' seen '  and  ' unseen '  methods for TraktTask to handle movie events . 
 Fixed bug where draining the queue could leave it with a non - empty size ,  causing incorrect behavior . 
 Added listener for onKeyShortcut (  )  to handle keyboard shortcuts
 Added updateExpandButtons (  )  method to NotificationContentView
 Fixed bug in handling file paths for XML files
 Fixed switch statement to use a more readable format . 
 Added test for handling of proxies in Relative302Test
 Added setItemAlignment (  )  method to RowsSupportFragment . 
 Added dot . wordpress . com URL to delete site domain in DeleteSiteDialogFragment
 Fixed bug in RemoteSCM handling for HistoryGuru
 Fixed issue with null pointer exception in OClientConnection . java
 Changed the logic to determine the suppressed parameter to only consider the type element of the parameter instead of the parameter itself . 
 Added fragment transaction for ReaderActivity
 Fixed issue with duplicate request registration
 Commit message :  Fixes issue with desktop user - agent string in ThemePreviewFragment to improve desktop preview of sites . 
 Added close (  )  method to deallocate resource leak in AbstractDnsMessage
 Added support for backpressure in `window` operator
 Fixed return type of getOres (  )  to match 1 . 8 API changes . 
 Added check for Windows platform in IdeFinder class
 Added directory path for local files in IOSFiles class
 Fixed stack trace logging in SocketUtils . 
 Added validation for unblockedRating in TvInputManager
 Added support for application media overlay sublayer . 
 Added ability to set fetch source in ExplainRequestBuilder . 
 Fixes adapter inheritance and stability issue
 Fixed issue with intentional failure in TestRequest . java
 Fixed formatting issue in AndroidTestRunConfiguration . java
 Fixed bug in cursor movement for MediaStore queries . 
 Added JSONException handling for magic link request error responses
 Fixed deprecated method signature in ErrorDispatcher . java
 Added support for animating heatmap tiles
 Fixes null pointer exception in SwiftInputStream . java
 Fixed timeout for nested creation in ByteBuddyTest . 
 Added method for cloning arrays in GwtPlatform for web - mode
 Added unsubscribeAllAnnotated method to AbstractViewModel
 Fixed encoding setting for MavenModuleImporter to ensure correct charset detection . 
 Added support for removing inline HTML tags in AtomFeedItem . 
 Changed how build number is initialized in PluginManagerCore to use a hardcoded snapshot value instead of null or a fallback mechanism . 
 Fixed buggy token removal logic in StorageService . 
 Added method to start Growl Notification service in GrowlNotificationActivator . 
 Fixed bug in HttpResponseChannel . java :  corrected handling of STATE_BUF_FLUSH . 
 Added new listener for onPatternStart (  )  in KeyguardPatternView . 
 Added check for correct width and height before calling onPositionChanged (  )  in RectangularShape . 
 Fixed issue with text contents for element type . 
 Fixed beta lower and upper bounds to ensure consistency . 
 Added volatile modification to lastUpdateTimestamp for faster expiration detection
 Fixed DuplicateFileException error message formatting . 
 Added getUserAuthId (  )  method to return users ID used for authentication
 Fixed binding issue for LifeLine server ,  now listening on port   +  port  +   . 
 Fixed bug causing incorrect calculation of breaker usage in multi - threaded tests
 Changed how non - blocking writes are tested to improve Linux compatibility . 
 Fixed formatting issues in ForecastFragment . java
 Commit message :  Fixes  # 123456 :  MidblockMatchState :  corrected calculation of position error and travel error . 
 Fixed issue with Cassandra stopping unexpectedly after test suite completion
 Added constructor to APNS class
 Added nullable `hostInfo (  ) ` and `serverStatus (  ) ` methods to `MongoStats` for better type safety . 
 Fixed suppress warning for unchecked cast in Iterator implementation
 Fixed timer for Netty client channel alive checks . 
 Fixed bug in prePollCheck (  )  method to correctly handle pollDirectory (  )  return value . 
 Fixed issue with missing umbrella element path for element :   % s
 Fixed bug in reduce method to handle maximum row size correctly
 Added getRotation (  )  method to Sprite class
 Added new style options page and updated rotation and colors tabs with improved layout and UI . 
 Changed test messages for better readability . 
 Smoothed heightmap functionality added / changed in WorldEditListener . java . 
 Fixed bug in handling too long text in ConsoleViewImpl . 
 Added const variable MAX_INT with value 0x7FFFFFFF and updated RIL_ERRNO_INVALID_RESPONSE to  - 1 . 
 Added ApplicationWindow interface to ConfigurationManager
 Added check for changed version map size and enabled gc deletes in onSettingsChanged (  )   ( diff  - r 4d03399 .  . 133f099 )  . 
 Fixed bug in Utils . delete (  )  to ensure proper file deletion on Windows  (  # 1134 )  . 
 Changed how local build rule outputs are calculated and cached . 
 Added method switchViewForContext ( Context )  to AbstractContextSelectToolbarStatusPanel . 
 Fixed bug in country sorting in GeoviewsModel . 
 Fixed issue with leaked window in StatsWebViewActivity . 
 Fixed null pointer exception in MainService . java by updating receiver registrations . 
 Added method isArray (  )  to ClassReflection class
 Added privileged method to retrieve legacy VPN information
 Added check for uninitialized via count in ViaInstruction class
 Added support for layout parameters in View . java
 Added subTarget to TrackingServiceTarget for better dependency tracking
 Fixed null or empty collection join issue
 Changed the PROFILER_AUTODUMP_INTERVAL configuration option to dump profiler values at regular intervals with a time expressed in seconds . 
 Added support for printing supertype in BeanSerializer
 Changed `FakeAddress` class to include `countryCode` field and its getter / setter methods . 
 Added quoted identifier for foreign key attribute in SQLForeignKeyManager
 Fixed issue with invalid FqName after unsafe check
 Fixed issue with callback handling in ResponseNotifier . 
 Added constructor for DerbyTorodbMeta with required parameters
 Fixed search order for Platform containers in DefaultPlatformContainer
 Added import for  @ Ignore annotation in ReferrerPushStrategyTest
 Fixed bug where untagged responses were not being handled correctly in ImapStore . 
 Fixed maxWaitQueueSize in Mongo . java to match changes in MongoClients . java
 Increased maximum message size to 16 MB for TestServiceServer . 
Added configuration option for network lock timeout in OrientGraphNoTxRemoteTest
 Added method isProjectOrWorkspaceFile to FileBasedIndexImpl
 Fixed endpoint for isWordPressCom method to correctly handle URL null values
Commits message :  Added import for class rule in JournalShutdownIntegrationTest . 
 Fixes issue with extra views not being properly inserted in StickyGridHeadersBaseAdapter . 
 Added config option to disable populating cache for CachingClusteredClient . 
 Fixed bug in alarm ID storage
 Fixed bug in DeleteReturnRouteMessageClass to properly handle invalid messages  (  # 17f52ef )  . 
 Added support for time zone in Calendar . getInstance (  )  in StandardConvertersTest . 
 Added XML_MIME_TYPE pattern match for proper file type handling
 Added new suggestion span for misspelled word in Editor . java
 Added support for service name with wildcard in register (  )  method
 Added new filter to skip danmakus older than 500ms or shown for more than 500ms  ( or both )  . 
 Added resource handlers for WebMvcAutoConfiguration
 Fixed catch block to handle ClassCastException . 
 Fixed generic type parameter for improved code generation
 Added import for org . neo4j . io . fs . FileUtils for better test reliability
 Changed method `updateInternal` to reset state information for the sketch object ,  including renaming recent files . 
 Improved InMemoryLookupCache implementation for faster vocabulary computation
 Changed how the `analyzeSupportedDatabases` method is called to match the updated implementation in the `b` branch . 
 Added focus management code to DirectoryFragment to improve search behavior . 
 Changed license terms from Apache License 1 . 0 to Apache License 2 . 0 . 
 Fixed serialVersionUID to match API changes . 
 Changed import statements to include new package names . 
 Added ability to expand / contract items on right - click in UITreeView . 
 Added new mappings for range and removed old mappings for range in LayeredLexerEditorHighlighter . 
 Added plus icon color resource  ( R . color . fab_plus ) 
 Added new method to mockJdk (  )  to return a mock JDK with a custom name
 Fixed method call order in GrClassImplUtil . 
 Added new calculation for frame time in milliseconds for smoother animation . 
Added a new method to the `MemberAttributeConfig` class in `hazelcast / src / main / java / com / hazelcast / config` to allow for the easy configuration of attributes for members . 
 Fixed bug in ViewPositionAnimator . java by updating fromClip . set ( fromPos . view )  to fromClip . set ( fromPos . viewport )  for correct viewport calculation . 
 Added BasicAuthenticator to TestRequest for improved security
 Fixed issue with localDeleteTime read validation in SuperColumn . java
 Added TextPaint for improved rendering of icon names
Improved exception handling in AbstractXMPPConnection and XMPPTCPConnection classes . 
 Fixed issue with childEnd being greater than childrenStart in TwoWayView . java
 Added check for links with multiple lanes in BinaryRoutePlanner . 
 Fixed visibility of filter radio group and swipe refresh layout in NotificationsListFragment
 Added support for changing text property in TextChange class . 
 Fixed bug where child item text was not being updated correctly . 
 Fixed issue with AnalyticsTracker ,  now tracking article opened correctly . 
 Changed inheritance many - to - many foreign key test to exclude disabled feature . 
 Added new SessionBindingHandler implementation to handle Session management in Ratpack . 
 Added support for sliding pane layout with customizable slide range and parallax effect . 
 Added DetachVolumesChaosType to the enabled chaos types in BasicChaosMonkey  ( commit 61274ef )  . 
 Changed how Docker container stopping is handled to account for missing containers . 
 Fixed method injection issue in ActivityWithInvalidMethodInjection . java
 Fixed bug causing incomplete attachment downloads . 
 Changed LexerActionExecutor field from nullable to non - nullable . 
 Added support for nested save points and enabled batch use transaction for SQLite Android database type . 
 Added missing method `disjoint` to `Sets` class . 
 Added ignore annotation to PeerPriority test due to flakiness . 
 Added  ' isDeferred (  )  '  and  ' isDynamic (  )  '  methods to TldAttributeDescriptor interface
 Fixed bug in JAASAuthenticationCallbackHandler array
 Fixed exception handling for loading DexFile . 
 Added syntactic sugar for the same RedirectConfig instance
 Added installerUid field to PackageManagerService class
 Changed how data is generated for the scatter chart to improve accuracy . 
 Added single - threaded updateUIExecutor for improved UI responsiveness . 
 Added QUIT command to SmtpTransport . 
 Formatted source code for portal - impl module . 
 Fixed issue with unread items group visibility . 
 Fixed broken pipe exception in serveResource method . 
 Added logging configuration for indices . recovery and action . search . type
 Fixed bug in LocationTest . java regarding serialization of Location objects
 Cached content of local files for improved performance . 
 Added method  ' removeTarget ( int )  '  to class  ' Transition '  . 
Added parent - specific artifacts to DependencyGraphBuilder . 
 Added new block location test for FileSystemBlockLocationIntegrationTest . 
 Added dispose method for WaterRipples class
 Added check box to select multiple comments in bulk
 Added recursive visitChildrenRecursively method to VfsUtilCore
 Added support for Ruby ' s `Class . new` syntax in `Call` transformation . 
 Added Map < Object ,  DBObject >  involvedObjects field to DatastoreImpl to store involved objects for merge operation . 
 Added method to replace  '  &  '  with  ' ~ '  in UI elements  (  # deprecated ) 
 Added retry mechanism for requestSpace (  )  method to handle failed RPC calls . 
 Added new method to ignore path of root directories in TaskFilePropertyPathSensitivity
 Added focus request to contact list panel in MainFrame . 
 Disabled proxy bypass for account ID  ' accountID '  in SipStackSharing . 
 Changed the `System . out . println` statements to print the current balance and receive address of the wallet before testing the fee level . 
 Added index definition for  ' name '  with value  ' index '  . 
 Optimized OptimizerNode class to keep unique properties for input  # 73bcc6dc3e . 
 Added new field token to TypePool implementation
 Fixed null pointer exception in beforeInvocation (  )  method
 Fixed encoding issue in TestUEncoder . 
 Added support for files in both srcs and hdrs . 
 Fixed shutdown method to properly shut down GraphDatabaseShellServer
 Changed foldLeft and foldRight methods to use ` ! ` instead of `` to avoid empty string concatenation . 
 Fixed issue with incorrect model node order in ParseAndMarshalModelsTestCase
 Fixed size thread pool for image loading improved from 4 to 8
 Added ReflectiveConfigOptionLoader . loadOnce (  )  to TitanFactory constructor
 Fixed checkValidTagParameters method to handle missing defaultImageResId ,  width and height parameters
 Renamed moveToStarted / moveToStopped methods for better naming consistency
 Added new location mode options and updated documentation for existing modes . 
 Added listener to update current sheet view height based on layout changes
 Fixes issue where TabLayout would not properly update selected tab position on rotation . 
 Registered resource roots with renamed class :  `RegisterResourceRoots` replaced `Register` . 
 Added test for Czech language with no XML rules . 
 Fixed null pointer exception in FileBackupHelper . java
 Improved timeout values for get ,  is ,  set ,  assert ,  enter ,  and click methods in Solo . java . 
 Added default layout for group with name  ' Portlet '  . 
Added method isBikeParkAndRide (  )  to OSMNode and updated isBikeRental (  )  to return true if the node has the  ' bicycle_rental '  tag . 
 Added extra SSID and BSSID to latency broadcast in API 17 compliance . 
 Added null safety checks for equals method
 Changed NetTest . java to use random port number and adjusted traffic class ,  TCP no delay and keep alive settings . 
 Fixed error handling for unsupported scheme in CordovaWebViewClient
 Changed the `Java` class to import Apache Commons Lang ' s `ArrayUtils` and `StringUtils` methods . 
 Fixed type safety in ProcessorDefinition . java by updating the return type of the splitter method . 
 Formatted Source code with improved indentation and closing curly brace alignment . 
 Added JobManager . create (  )  method to reschedule jobs in JobBootReceiver . 
 Added replaceExtras method to ActivityManagerService to update requestIntent . 
 Fixed estimation of training time in NNModel . java
 Added SearchQuery implementation for Hibernate Search
 Fixed bug in VcsLogRefresherTest :  corrected handling of null log provider
 Changed ActivityStackSupervisor . java to throw SecurityException for embedded activities without allowEmbedded flag . 
 Added new CodeFoldingOptionsProvider to the panel . 
 Added printer index selection to PrintActivity
 Fixed bug in SkyframeBuildView . java by updating the labelToLoad variable to correctly handle file symlink cycles . 
 Added visibility check for download progress and duration in NewEpisodesListAdapter
 Added empty array for IncludeInfo to improve performance . 
 Changed default implementation of `canRun` to include additional checks for coverage . 
 Added method call to siteNodeChangedEventHandler for improved performance when not on dispatch thread . 
 Added print statements to log HADOOP_VERSION and file pathnames in BaseHadoopTest . 
 Added check for first edge being a street edge in GraphPath . java
 Changed the implementation of `isExtendedClassNamedAsException` to correctly handle nested extends clauses . 
 Added check for processor type existence in Pipeline constructor
 Update DozeService :  added requestPulse (  )  call to register for sensor updates
 Added Daemon thread attribute to ThreadCpuStats collector for better performance
 Added loadXML method to load XML files in PApplet class
 Fixed issue with loading internal DTD file . 
 Added chained annotation reader for reading owner type . 
 Converts multi - level sensor values for temperature scaling . 
 Changed behavior of JpaTraceEventMessageTest to use JpaTraceEventMessage . 
 Fixes issue with cursor windowing
 Added method handle for coercion from   +  Type . getTypeName ( type )   +   to VARCHAR . 
 Fixed issue with Toast notification in onFailure method . 
 Added check for mainJarName in isInstallationHome method
 Added array argument types to MetaObjectProtocol . java for more informative method calls
 Changed method signature for interface PaintingParent to include `Rectangle getChildRec ( Component c ) ` method . 
 Changed method to include file name suffix in MIME encoding . 
 Fixed bug in completedBytes calculation
 Changed how `ModuleRootManagerImpl` handles `RootModel` to improve performance . 
 Added fireWriteComplete method to LocalChannel class
 Added new method to return a new Bitmap with the native delegate . 
 Modified RecoveryBackwardsCompatibilityTests to use less aggressive settings for BalancedShardsAllocator . 
 Added new line of code to handle dirty regions in HardwareRenderer
 Fixed issue where nextRow could be greater than lastRow in some cases
 Fixed bug in handling of waiting APNs in GsmDataConnectionTracker . 
 Added Toast . makeText (  )  method for displaying toast messages on UI thread . 
 Fixed table name in ScanRequest based on class configuration
 Added method  ' outputBaseTime '  to log base time in milliseconds since absolute start time  ( epoch ) 
 Added custom named factory class with new name
 Changed method signature to throw BlockStoreException instead of re - throwing it . 
 Fixed DNS resolution issue in ServerDialback . java
 Changed method `isGoModule` to also check for the presence of the `GO_VERSION_FILE_PATH` file in the module ' s root directory . 
 Fixed compatibility issue with Android 11 . 
 Added new import for Spring Core IO ' s ClassPathResource
 Added new transaction handle implementation for improved performance
 Changed Sketch . java to use `setCode (  ) ` instead of `setCode (  ) ` and `rebuild (  ) ` and `repaint (  ) `
 Fixed highlight ranges for improved readability . 
 Added colorized presentation for fields
 Added new executor service for image resizing . 
 Changed `mMagnificationSpec` field in `DisplayContent` class . 
 Fixed spoofing behavior in GangliaReporter to correctly identify the host label . 
 Added concurrent map for casts in BeanPath
 Changed `return false` to `return true` in `parse (  ) ` method . 
 Added new method `set` to return a Set instance containing the provided items . 
Update VisualTraverseVisitor to use a sleep interval of 50 milliseconds instead of 10 milliseconds to avoid rapid edge highlighting . 
Update FormatterBasedLineIndentInfoBuilder to use PsiDocumentManager . commitDocument instead of manually committing the document . 
 Added new field  ' rollingCountFallbackMissing '  and updated existing fields . 
 Added method closeTransactionConnection ( PreparedStatement ,  Connection ,  boolean )  to rollback / commit transactions
 Commit message :  Fixes issue with incorrect total mapped bytes count in MappedFile . 
Added import statement for ReplicaSet category in DBTest . java
 Fixed face detection for full - size images in PhotupImageView
 Changed how height is calculated for the LanguageConsoleImpl class . 
 Added support for providing ranges for multiple elements in RegExpRangeProvider . 
 Added support for Java - Kotlin modules in KotlinWithGradleConfigurator
 Added fireEvent (  )  method to persist configuration changes for DBNDataSource . 
Added new pattern to disallow PPs that are next to a CC or CONJP ,  and modified existing patterns to only allow PP  <  PP combinations if there is no conj ,  verb ,  or other pattern that matches pcomp under it . 
 Added MotionEvent handling for dismissing popover on touch events . 
 Added nullability check for personal field in Address constructor
 Added  ' useWideViewPort '  and  ' loadWithOverviewMode '  settings to improve web view performance
 Fixed bug in DeepWaterFrameIterator compute2 (  )  method by changing Key rowKey creation to include frame size for better cache optimization . 
 Added single argument constructor for IgnitedAsyncTask
 Added logging statement for caught exceptions in ONetworkProtocolHttpAbstract . 
 Fixed issue with FragmentManager ' s Activity . onDestroy (  )  call . 
 Fixed issue with closing input stream in AfcClientTest . 
 Fixed issue with remote connection closure handling in WebSocket . java
 Added check for directory scanning progress indicator
 Fixed null pointer exception in ExifInterface constructor
 Added constructor parameter validation for payload and headers
 Added support for loading truststore and keystore from classpath in SslUtils
 Added test class for FeatureMethodEarlyBeanProxyTests
 Fixed log statement to include request method and URI . 
 Fixed whitespace issues and improved performance by removing unnecessary semicolons and whitespace . 
 Implemented new logic for filtering VCS results . 
 Added cache for Sdk with project ID  ' project '  and Sdk ID  ' sdk '  . 
 Added 14 new blocks to the storage ,  including file size :  305 bytes . 
 Added new layer update mechanism for TextureView
 Fixed type safety issue in LocalActorRegistry by adding throws SuspendExecution to the tryGetActor method . 
 Changed method signature in `ApplicationImpl` to use `Thread . interrupted (  ) ` instead of `ReflectionUtil . resetThreadLocals (  ) ` . 
 Added loading of exception rules for  ' an '  and  ' a '  from external files
 Fixed bug in value container handling . 
Added SystemIndependentName to file paths in IncArtifactBuilder for better compatibility
 Changed how exceptions are handled in `FileProcessingCompilerAdapterTask` . 
 *  `diff  -  - git a / core / src / com / google / zxing / qrcode / detector / FinderPattern . java b / core / src / com / google / zxing / qrcode / detector / FinderPattern . java` :  Added support for finder patterns in QR Codes
 Fixed error message for DatePart function
 Fixed null pointer exception in World . destroyFixture (  )  by removing unnecessary user data setter . 
 Added convenience method to register an action  (  # ce5a6d3 )  . 
 Fixed  < clinit >  method implementation for ClassDefinition . 
 Fixed bug in HttpRequestBuilder to correctly handle empty param lists
 Changed how `Location2IDIndex` is imported to avoid unnecessary dependencies . 
 Fixed issue with empty / incomplete property file for RACKDC_PROPERTY_FILENAME . 
 Fixed timestamp comparison in Result class to use isEqual (  )  and getZone (  )  . getOffset (  )  for correct time zone handling . 
 Fixed null pointer exception in WorkspaceFileFunction . java by adding null check for workspaceFileValue in compute method . 
 Added Timber . tag (  )  for custom event logging
 Added method overloads for writing multiple bytes . 
 Added method onPreMessageReceived to PeerEventListener interface
 Added import statement for android . annotation . SuppressLint ;  updated OPENED_FROM_DRAWER_DELAY constant to 250ms ;  updated  @ Override annotation for onCreate method . 
 Changed method name from sendLoginEmail to auth / sendLoginEmail in RestClientUtils . java . 
 Converted fireOnTestIgnored method to fireOnTestIgnored for SMTRunnerEvents . 
 Fixed constructor injection in ConstructorInjectionTest
 Added SelectionEndOffset to CompletionInitializationContext for improved insertion experience . 
 Added method to reset statusReceived flag in NettyResponseFuture . 
 Added support for cloning MacHashFunction instances
 Changed the test timeout from 500ms to 60s in JDBCExtractionNamespaceTest . 
 Changed directory path in OSMReaderTest from  / tmp / OSMReaderTrialsTest / test - db to  .  / target / tmp / test - db . 
 Fixed logging statement to improve debug output
 Added new synchronized static method to getStringManager for faster lookups
 Added new buffering - related events to MediaPlayer . OnInfoListener . 
 Fixed issue where DropTypeStatement was incorrectly checking for used keys in MapType . 
 Added import for org . junit . Ignore annotation to ignore a test due to incorrect method calls in logging . 
 Added method  ' awakenScrollBars (  )  '  to AbsListView
 Added UI widgets for game creation and saved games menu . 
 Changed NotificationGroupManager to include the notification in the summary string . 
 Commit message :  Loaded data into WebView using  ' data '  scheme URL . 
 Added method to set the system UI visibility and added a listener for changes to the system UI visibility . 
 Fixed dependency cycle issue in Loader . java
 Added method equals (  )  override for Versioned class
 Added field  ' minorCount '  to GarbageCollectionMetricSet
 Fixed issue with child view performClick (  )  method
 Added relationship creation for Trinity in ShellTest . java
 Changed behavior of `LaterInvocator` to wrap exceptions instead of re - throwing them . 
 Fixed type check for RObjectEntry view to correctly handle  ' xts '  type . 
 Added debug logging for value and formatted value in BindyCsvFactory
 Fixed bug in BackpressureUtils . java by updating requested field to Long . MAX_VALUE for all requests . 
 Fixed import statements for better compatibility . 
 Commit message :  Fixes timer wheel cancelation logic to be idempotent . 
 Changed zoom limit from 22 to 25 in MapActivity . 
 Added apply (  )  method to Material class to update material parameters . 
 Added support for cherry - picking commits with the new `cherryPickHead` file . 
 Fixed issue with empty lines in FileTypeUtil . java
 Logged request :   [ GET  / my - index ]   [ 200 OK ] 
 Added hasDrawnLw (  )  method to WindowState class for improved replace window logic . 
 Added debugging log statement for publicNotification in BaseStatusBar . 
 Changed `value (  ) ` default value from ` / a` to ` / z` in `CustomAnnotationTest` . 
 Added BlazeRuntime . java :  allowedMissingInputs field to handle missing action inputs from merged Blaze modules  (  # 733ecfa )  . 
 Added constructor for Sticker entity
Added warning for  ' this '  reference in arrow functions inside object literals  ( Es6RewriteArrowFunction . THIS_REFERENCE_IN_ARROWFUNC_OF_OBJLIT ) 
 Renamed method  ' setIpAddress '  to return a new DhcpResults object for better handling of lease renewal and failure notifications . 
 Added exception handling for invalid input lengths in StringToCharacterConverter
 Added tie breaking logic for getMatchingParams to handle cases where multiple methods match
 Fixed constraint name in SQLiteBooksDatabase class
 Added test for failed randomly ,  a candidate to ProblematicTest
 Added new disable modes for keyboard shortcuts
 Fixed issue with  @ JsonIgnore annotation not being properly ignored in ModelSerializerTest . 
 Update MediaLocalToRemoteId method added
 Remove connection from dispatcher and remove from connections list
 Fixed session Id validation to prevent session fixation vulnerability . 
 Fixed cancelRequest method to use  ' cancelExistingRequest '  instead of  ' cancelRequest ' 
 Added new field  ' normalized_number '  and updated type of  ' number '  to TEXT . 
Added testLockConditionSignalAllShutDownKeyOwner (  )  method in LockTest
 Added support for type parameters in PrettyCommentsPrinter  (  # 759 , 7 ) 
 Added logic to handle more indices than the configured maximum in AbstractIndexCountBasedRetentionStrategy . 
 Added import for io . undertow . testutils . AjpIgnore
 Added  ' mStructureChanged '  field to track when layout structure has changed . 
 Added name of the class to the memory writer
 Added default value of `mainThreads` to 2x number of CPU threads for improved performance
 Update SearchController to use minute interval for histogram
 Added finally block to resetConfigOverrides (  )  in DropwizardTestSupport . 
 Changed the implementation of `DateMidnight` to use a `YearMonthDay` instead of an `Interval` for more suitable representation of the millisecond instant at midnight . 
 Fixed bug in findNextWordEnd (  )  to correctly handle cases where the next word starts with a digit
 Fixed bug in GearLoadingRenderer to display correct number of gears . 
 Changed a return statement in the `ensureForegroundActivity` method from `ActivityManager . PROCESS_STATE_TOP` to `ActivityManager . PROCESS_STATE_TOP_SLEEPING` . 
 Changed Clicker class to use the new `View` parameter instead of the old `View` variable . 
 Fixed issue with SVG document parsing and added support for SVG headers . 
 Added test for executeScriptWithLineBreaks
 Added createTableObservers for nodes and edges in AppearanceUIController
 Added optimization for task creation in ActivityStarter
Fixed issue with dropping stash after conflicting merge  (  # 1234 )  . 
 Added method `customView` to inflate custom layout for MaterialDialog
 Changed how container volumes are described in the help message to include the full path for both host and container paths ,  and added the ability to specify a read - write or read - only volume using the suffixes  ' rw '  or  ' ro '  . 
 Fixed output mode to INLINE for compatibility with MongoDB 4 . 2
 Rebuild inheritance and set linked classes for OConsoleDatabaseImport
 Added handling for escaped characters in literal expressions . 
 Added producer cache to XmppRouteTest
 Added method isTranslucentNavigationEnabled (  )  and setTranslucentNavigationEnabled ( boolean ) 
 Added test for issue  # 6819 :  Ignore whenBlock_whenNoSpace (  )  method when topic capacity is reached
 Added new line for installing 2 in ManagementCommunicationService start method
 Fixed syntax error in JavaPredefinedConfigurations . java
 Fixed issue with contact not being in roster when starting conversation . 
 Added import for FastAdapterR . selectableItemBackground for newer Android versions
 Added support for time zones to ZonedChronology . 
 Changed assertions for max outer concurrency to check if availableProcessors (  )  is greater than or equal to 4 ,  to be more deterministic . 
 Added import for com . intellij . util . containers . OrderedSet and updated method to use it in getExternalLibraries (  )  to improve performance . 
 Added protection against XSS attacks in HTTP headers . 
 Added new quickfix for cleaning up unused imports
 Added remote caching notification for worker . 
 Fixed compatibility check for attachment provider . 
 Added new WebDeploymentActivator method to handle deployment processors for web applications . 
 Fixed issue with logging message for failed file text retrieval . 
 Fixed typo in package name in ChunkGeneratorManagerImpl class . 
 Added common case for 1 table scan per map - work ,  and rare case for some joins
 Fixed connection test query to improve test accuracy . 
 Improved performance for column iteration in SimpleSliceReader . 
 Added border to JBScrollPane
 Added onItemUse method to StripesHandlerPlaceBlock to call the original placeBlockAt method with a new parameter . 
 Changed the import for ModelTestUtils to add the new import . 
 Fixed timeout value in Parser class from 5000 to 10000 . 
 Added shortcut set for  ' Diff . ApplyLeftSide '  and  ' Diff . ApplyRightSide '  actions in TwosideBinaryDiffViewer . 
 Changed import for `JsonDeserializer` to `com . fasterxml . jackson . databind . JsonDeserializer` . 
 Fixed issue with DeadException . md file generation in BugPatternFileGeneratorTest . 
 Changed how `verify` methods are called on `MessageManager` to match the new implementation . 
 Added method mTextDisplayLists [ blockIndex ]  . isDirty  =  true ;  to handle dirty text display lists in dynamic layouts . 
 Added default value for  ' Id '  field in Mongo interface
 Added check for table existence before attempting to create alias . 
 Fixed bug in Cassandra keyspace initialization in ZuulFilterDAOCassandra
 Refined callback interface and added extra callback methods for background thread to call . 
 Added top padding to StaggeredGridView for improved layout performance . 
 Changed how ACTION_DOWN events are logged for panel widgets . 
 Fixed issue with collapsible if statements in JMXConfiguration . java
 Changed `show (  ) ` method to return a `MaterialDialog` instance instead of displaying it directly . 
 Renamed SimpleGraphServiceImpl to GraphServiceBeanImpl with 90 %  similarity index . 
 Fixed issue with Windows  ' start '  command dropping anchors from URLs for local URLs . 
 Added method `auxAllocateAndAdjustFields ( long ,  long ,  long ,  long ) ` to `HashSlotArrayBase` for better error handling . 
 Fixed issue with transaction inputs not having non - default sequence numbers for lock times to work correctly . 
 Changed CommandLineOptions to print manual page on standard out . 
 Added final static variable for LOOP_INC_VALUE in LoopStep . 
Update TaskManagerConfig to use 1 - second info update interval and 1 - second status refresh max wait
 Added equality check for bounds in screen and layer in AccessibilityWindowInfo . 
 Added support for indeterminate progress bar in AccountActivity
 Added check for null graph element before selecting next fragment
 Added boolean  ' positive_response '  field to CreateFrame class for improved regression performance
 Added layout direction setting for title view in PhoneWindow
 Added ability to set focusable window state for FloatingDecorator
 Added support for new ABIs in Build . java
 Added LinkedHashMap for function arguments
 Added support for serving Markdown files as HTML . 
 Added Kuromoji analyzer to integration tests
 Added FOCUSABLES_TOUCH_MODE to the list of focusable modes
 Added visibility check for gitignores in NewRepositoryPage
 Commit message :  Fixes issue  # 1234 by updating the constructor of `ProducerFailedException` to call `super ( cause ) ` instead of `this . cause  =  cause` . 
 Fixed typo in event name for  ' Reader  -  Loaded Search '  instruction . 
 Commit message :  Fixes ruler control layout issues in night mode . 
 Fixed exception handling in Loader . java to correctly log and handle errors . 
 Changed behavior of AuthorizedClientUser in FileSystemMasterIntegrationTest to use null instead of empty string when starting a concurrent thread . 
 Added dirty text range check for External Tool Pass . 
 Changed the default implementation of the `getChargedThroughDate` method to use the `chargedThroughDate` field instead of the `clock . getUTCNow (  ) ` date if it exists and is less than the current date . 
 Fixed buggy rendering of task execution tree table cells . 
 Added clone method for _numOffsets to preserve original values
 Changed handling of unknown scope in ElasticsearchIntegrationTest . 
 Added exception handling for unknown host exception in Response class
 Added new line character limit for log messages in AnalyticsTree class to improve readability . 
 Fixed bug in handling cluster URL in VoldemortBuildAndPushJob . 
 Added MachineReadingProperties . percentageOfTrain field to keep percentage of train data for learning curve experiments . 
 Added configuration injection for BulkLoaderVertexProgram
 Added import for InvalidRecipeException in MachineManager . 
 Added support for reconfiguration in Bitmap . java
 Fixed naming convention for MongoDB storage provider class . 
 Added 200ms delay after scroll down to allow for full view rendering
 Rewritten generateBigTree method to use correct coordinates . 
 Fetched config with  ' build '  variable and updated it with value  ' dev '  . 
 Added toString (  )  method for CarrierIdentifier class
 Changed method signature for `eventCodes (  ) ` and `eventBufferSize (  ) ` to return `this` instead of the original return type . 
 Fixed deprecated methods in WebIconDatabase . java and WebSettings . java
 Added support for intersection and union operators in the partition builder
 Fixed license year in StatusResultMatchers . java
 Added check for null MAX_INACTIVE attribute value in IdleResourceInterceptor . 
 Added import for ProgressManager in PyResolveUtil . 
 Added preference for notification ringtone . 
 Fixed issue with handshake synchronization in SslHandler . 
 Added getTraceStream (  )  method for tracing lexer output
 Added method  ' replaceUnicodeDigits '  to normalize Arabic / Unicode digits in PhoneNumberUtils . 
 Added method to setUsePooledMemory in DefaultAsyncHttpClientConfig
 Fixed issue with PIN input ,  now prompting for PIN before requesting access token . 
 Update KeyProcessorContext to use new data context and focus owner . 
 Fixed compilation issues with Java 14
 Fixed issue with empty response body in TokenEndpoint . 
 Added empty CacheManager to UncachedSpiceService for requests not to be cached
 Added recent project paths for opened projects . 
 Added new CDMA roaming mode preference . 
 Changed method signature in LightIdeaTestCase to exclude unnecessary parameter . 
 Fixed unused import statement in InnersOnDemand1 . java and InnersOnDemand2 . java
 Fixed annotation check for Jersey provider
 Improved performance of detecting indentation on hot files in JavaAutoDetectIndentPerformanceTest . 
 Added GeoShapeFieldMapper to  ' location '  field in GeoShapeFieldMapperTests
 Changed how the parser tree is processed to correctly handle invalid parse tree shapes . 
 Added commit and compact database methods to flush existing data and reduce file size . 
Added ability to specify document types in ValidateQueryRequest
 Fixes ClassLoaderLock to wait for child lock instead of parent lock when trying to acquire lock
 Fixed clone method to avoid exception
 Added new constructor for LineExtensionInfo with text and attributes parameters
 Added constructor for RequestBuilder with disableUrlEncoding and validateHeaders options
 Changed how the error message is handled to properly identify the expected exception . 
 AbstractTreeTableModel :  updated to remove unused import
 Changed the `tzInfo . toString (  ) ` call to `tzInfo . toString (  ) ` in the `daylightSavings` field . 
 Added support for custom evaluators for types not automatically understood by the animation system . 
 Added serialized form for FailProcessorException
 Added new method to readTimeout parameter in AjpProcessor class
 Fixed bug in UserValue . isString (  )  method by changing return type from boolean to boolean . 
 Added import statement for ConsoleEvent in Build . java
 Update AioSocketChannel to ensure readerIndex is 0 before async write to prevent unexpected buffer modification
 Added method to set maximum wait time in milliseconds  (  # 5fcb1518a )  . 
 Fixed layout issues in TagGroup . java
 Added jitter to compaction interval for better database performance
 Added automatically quotes to argument in ExecUtils and modified RunCommand to add  -  - conf and  -  - instances options . 
 Added method invocation for Result return type in RouteBuilderImpl . 
 Fixed regex pattern for path splitting in UriMatcher . java
 Fixed bug in token update logic
 Added new kernel class ( es )  and method ( s )  to RootbeerScene . 
 Added getActivity (  )  . getIntent (  )  . putExtra ( TOKEN_ID ,  model . getId (  )  )  ;  to save task ID in intent for next screen
 Fixed issue with empty search path in DefaultGroovyPageLocator
 Upgraded SvnConfiguration to version 16 ,  added new state and authentication manager fields . 
 Added setKeyedTag method with improved parameter naming for better code readability . 
 Added new package names to Util . java
 Fixed layout compatibility issues in PlaybackControlsRow . 
 Fixed issue with exposed headers in CrossOriginFilter . 
 Fixed bug in handling of unboxed types in RedundantCastUtil . 
 Fixed bug in material cloning by correctly resetting bone matrices . 
 Added import for Android logging library
 Added import statement for Parcel class and updated encryption status field in PptpProfile class . 
 Fixes layout calculation for vertical layouts by removing unnecessary padding . 
 Added log4j logging for SnitchProperties class
 Fixed issue with handling of doc level references in CollectInputSymbolVisitor
Refactoring Constructor in C2 class
 Renamed S3UnderFileSystem . java :  added new method to open file at position
 Fixes issue with cropped image not being restored correctly ,  improves performance
Update ConvertBodyProcessor ,  SetHeaderProcessor ,  and SetPropertyProcessor to handle exceptions more robustly . 
 Fixed bug in email alert handling by updating backlog size calculation . 
Added getAppVersion (  )  method to Version class and used it to display the app version in the MainMenuActivity . 
 Fixed array escaping in PostgresArrayEscapingTest . 
 Changed Tuple interface to require equality and hashCode implementations . 
 Upgraded DatabaseHelper to version 51 with new settings for set install location UI . 
 Added support for handling OnErrorFailedExceptions with a cause . 
 Commit message :  Fixes issue with retaining records and objects in OIntentMassiveInsert . 
 Added import for Interceptor to StatefulSessionComponent
 Added new implementation of `force (  ) ` and `invoke (  ) ` methods in `Delay` class . 
 Fixed buggy code in OpenSSLEngine . java by removing unnecessary comment . 
 Fixed issue with remote connection delete database method ,  now using new OServerAdmin API . 
 Added pagination support to PaginatedRepository
 Changed how coverage information is computed for packages . 
 Added optimization correction check for lazy parseables in PsiBuilderImpl . 
 Commit message :  Fixes issue with LegacyBatchIndexApplier reusing appliers for different transactions in the same batch . 
 Added new page registration for  ' mytickets '  and updated existing registration for  ' LuceneSearchPage '  . 
 Added instance variable to hold the single instance of Main
 Added explicit name check for POJOPropertyBuilder . 
 Added option to ignore case sensitivity for filters  ( except file filter )  in GitLogProvider . 
 Added method `putClientProperty ( String ,  Object ) ` to `GraphSettings` interface . 
 Added test for IBM vendor in SignatureTests class
 Added null safety checks for commandLine in ScriptRunnerUtil
 Added assertions for chunk class comparison in NewVectorTest . 
 Added `setSelectedIndex` method to support indexing into a list of options . 
 Fixed bug  # 123456 :  Fixes issue with hidden TV input
 Added permission check for BLUETOOTH permission in BluetoothService . java
 Added permission flags for ACTION_OPEN_TREE and ACTION_OPEN_COPY_DESTINATION actions
Update query to use DISTINCT instead of SELECT  *  to improve performance  (  # 1234 )  . 
 Added method to filter available intentions based on hint
 Added bridge modifier to method  ' overriddenMethod ' 
 Fixed bug in HTTP2Session . process (  )  method
 Commit message :  fixed issue with expired tokens
 Improved AbstractMaster . java with new AsyncJournalWriter field . 
 Fixed activity reference in ActivityUtils to improve performance
 Update TestCase . java to use MongoDB server at port 27017 instead of 27018
 Changed implementation of PyClassImpl to improve performance . 
 Added  ' mFirstShow '  variable to track if the keyguard view has been shown for the first time . 
 Fixed method signature to match updated TestProgressEventVersion1 interface . 
 Fixed bug in Stopwatch . createStarted (  )  method . 
 Added test for empty objects being equals
 Added new command for curl to use when deploying solo containers
 Fixed issue with serialization of mock objects
 Commit message :  Added support for fancy fullscreen on OSX . 
 Fixed exception message in OtrChatListener to better handle encrypted message decryption errors . 
 Added support for JSON serialization of HystrixRequestEvents with improved formatting . 
 Added option to show horizontal scrollbar in LanguageConsoleImpl . 
 Added ability to expand / collapse nodes in Inspect Debugger Tree . 
 Fixed lambda value for GLMBasicTestRegression regression test . 
 Added close (  )  method to leaderLatch in DruidMaster to prevent leaks . 
 Added check for page size to be greater than 0 in PagingPredicate constructor . 
 Added new rule to stream with ID  ' streamid '  . 
 Added center layout for scroller in GeneralConfigurationPanel . 
 Fixed output of  ' System . out . println '  in LightInspectionTestCase . setUp (  )  to avoid inspection warnings . 
 Added validation of ticket granting ticket ID in CentralAuthenticationService interface  (  # 4871c20d54c .  . 699a39bb5cd )  . 
 Added method  < code > train (  )  <  / code >  and  < code > valid (  )  <  / code >  to  < code > Model <  / code >  class for improved API experience . 
 Added new transport stop names to the layer . 
 Fixed issue with binary representation of large integers by removing unnecessary bit shifts and using a more efficient modulus calculation . 
 Fixed bug in fireTrackerEvent (  )  method to correctly track search events . 
 Fixed output type for writer in Json . java
 Fixes visibility bug in task stack views by disabling reusing until it ' s fixed . 
 Fixed exception message formatting in SearchingForTestsTask to improve error logging . 
 Fixes issue with border image target in UIDisplayContainer . 
 Added method to remove messages for switch to longpress event . 
 Changed the import for `Arrays` from `edu . emory . mathcs . backport . java . util . Arrays` to `java . util . Arrays` . 
 Added focusable property
 Fixed bug in onCapturePartial / onCaptureProgressed method to properly handle closed camera device
 Fixed issue where no proxy - granting tickets were being processed for single logout . 
 Added  @ JsonIgnore annotation to field  ' type '  in Event class
 Added license header for PostgreSQL93Dialect
 Added new suppressor for missingRequire warning
`Added tests for file in - memory percentage calculation`
Update to use private inner class for thisT (  )  method to avoid warnings
 Added import for  @ PublicEvolving annotation to StoppableFunction interface . 
 Added ability to compute class reference in FieldDescriptorImpl
 Fixed issue with hide until date formatting
 Fixed issue where library version was not properly initialized in RepositoryLibrarySynchronizer ,  now using RepositoryLibraryProperties to get the library version . 
 Commit message :  Fixes JPAMockSFSBContainerTestCase test to use correct classloading mechanism . 
 Implemented TLS support in OkHttpClient . 
 Fixed filename method return type to match the updated implementation . 
 Reduced disable timeout from 5000 to 20000ms for BluetoothTestUtils . 
 Fixed issue with incorrect file listing in CommonQualityAssuranceTest . 
 Added water and flowing water to soft block checks
 Added static class BoundColumn for better performance
 Fixed bug where removing a view with the same name as a existing view would cause a NPE
 Added method to return local anchor B in RevoluteJoint class
 Added ability to merge override / implement members in OverrideImplementUtil . 
 Changed the position of the actions panel for watches in variables . 
 Changed how `loadText` is called to improve performance . 
 Fixed bug in InitialAccountRegistrationFrame to correctly handle group name comparison . 
 Fixed issue with invalid schema syntax in UseHandler . java
 Added method  ' getCompactionThroughputMbPerSec (  )  '  and updated method  ' setCompactionThroughputMbPerSec ( int )  '  in StorageServiceMBean . 
 Added final modifier to tvals field for better performance and consistency
 Added cancel pressed functionality to BreakAddEditDialog
 Added listener to set web . xml configuration when context starts . 
 Changed calculation of the soft clustering coefficient to use `zSum (  ) ` instead of `zProduct (  ) ` . 
 Changed how long the `queryNotifyLatch` waits for in the `ServerManagerTest` class from 25 milliseconds to 0 milliseconds . 
 Changed ActivityTest . java to include a new `setupTestingRoots (  ) ` method . 
 Added new static final string  ' VENDING_PROMO_REFRESH_FREQUENCY_MS '  for frequency in milliseconds to cycle through promoted applications on home screen or categories page . 
 Changed how IOException is thrown when storing state ,  improved error message . 
 Improved performance of the `ObservableCollapser` by adding additional parameter values for the `numToCollapse` and `numResponsesPerArg` methods ,  fixing issue  # 865 . 
 Added support for collecting up to 120 attributes in Value objects  ( including inherited )  . 
 Added constructor parameter for writeConcern in MongoDbClient
 Added new method to test maxBsonObjectSize in ReplicaSetStatus . Node
 Added negotiation type for plaintext in TransportBenchmark
 Added setSecondaryToolbarVisible ( false ) 
 Fixed exception message in GradleDistribution to include the default file passed . 
 Added method to retrieve list of lineage information
Added cleanup method for LaunchPlugin  (  # 38 , 4  -  >   # 37 , 9 ) 
 Fixed loading of util packages to use correct base package . 
 Fixed indentation in Capsule . java for improved readability . 
 Fixed bug in batch configuration enable property retrieval . 
 Added support for mapping SQL types to Java types ,  including updated mappings for TEXT ,  CHAR ,  NCHAR ,  and added support for CLOB and NCLOB . 
 Fixed bug in handling missing channels for request  # {} . 
Added JBPopup hint functionality to DocumentationComponent  (  # 1 ) 
 Added predefined selection for ListView in DrawerBuilder . 
 Added ConcurrentSkipListMap for recursive computeIfAbsent in ComputingTest
 Fixed issue with stopping worker in LocalTachyonCluster
 Added button panel to MainFrame for call history . 
 Fixed border size for ComponentTree component . 
 Added new AuthFutureCmd method to handle authorization grant requests . 
 Fixed issue with setting last modified timestamp for downloaded files . 
 Added method getCollectedCount (  )  and updated timeout handling to improve performance
 Fixed issue with FileTabCharacterCheckTest . java :  added missing closing parentheses . 
 Changed type of `Endpoint` variable from `org . apache . camel . Endpoint` to `org . apache . camel . spring . config . RouteBuilderRefTest` . 
 Rewritten TagViewFragment to add custom filtering logic
 Commit message :  Fixes issue with incorrect JSON parsing in JsonHttpResponseHandler . 
 Added ToolWindowManager instance to activate editor component
 Fixed issue  # 1263 by ignoring MessagingException when closing mailbox folder . 
 Added null check for unbinder in `onDestroyView (  ) ` to prevent leaks
 Added method signature for exitActual (  )  in PApplet . java
 Added edge iterators for OSQLFunctionDijkstra . 
 Added null check for urlString in getHttpRequest method
 Fixed ReachedMaxSizeException assertion in ClientMapStoreTest . 
 Renamed OilPopulate . java to OilPopulate . worldgen in common / buildcraft / energy . 
 Added locale - based uppercase transformation for PagerSlidingTabStrip
 Added new `nudgedTooFarRight` and `nudgedTooFarDown` variables to handle cases where the QR code is not fully visible in the image . 
 Added isolated UID check for privacy management in ActivityManagerService
 Added test for SplitSyncUpOperation in OperationTestSuite
 Fixed formatting issue in log statements for better compatibility . 
 Added testNestedStructure method to DOMGeneratorTest
 Added VDBG log statements to NativeDaemonConnector . 
 Fixed NPE in ApplicationImpl . assertReadAccessAllowed (  )  by adding missing braces . 
 Fixed compatibility issue with latest Android SDK . 
 Added editor tab placement configuration for Switcher action . 
 Added new enumeration values for the `T` enum in `Tokens . java` . 
 Fixed bug in detach method
 Added override for updateTextBounds method to include stroke width in text bounds calculation . 
 Added  ' JOIN '  keyword to SQL keyword list
 Fixed issue with packedAngleCos variable calculation
 Changed how `transportClient` is handled in `onModule (  ) ` to improve compatibility with Elasticsearch 7 . 
 Added newStreamException method to handle internal error in HTTP / 2 streams
 Added method  ' saveSettings (  )  '  to SiteSettingsFragment
 Added validation for number sequence in RequestArguments class . 
 Added clone method to RenderTarget class
 Added support for ISO 3 language code in HdmiCecLocalDevicePlayback . 
 Added logging configuration for testing . 
 Changed the layout of the ContactListTreeCellRenderer to add a new icon and label for instant messaging status . 
 Refactored `PyClassRefactoringUtil` to improve readability and reduce code duplication . 
 Fixed unregisterReceiver (  )  call to prevent IllegalArgumentException
 Fixed bug in RecommenderMapper to correctly handle empty top items list
 Fixed issue with shutdownDone not being marked as successful in writer thread . 
 Added new Derby database  ' dstest '  for testing purposes
 Added method to createMessage ( byte [  ]  content ,  String contentType ,  String contentEncoding )  for sending arbitrary MIME - encoding content . 
 Added new method  ' onPreviewShowed (  )  '  and updated existing method  ' onShowPreview (  )  '  to handle animation more efficiently . 
 Added check for bouncer showing before unlocking fingerprint . 
 Fixed calculation of average put time to avoid division by zero . 
 Fixed bug causing incorrect logging for cache property changes in DefaultPersistentDirectoryCache . 
 Fixed issue with empty list visibility in PeopleListFragment . 
 Fixes IllegalArgumentException when setting start delay to negative value
 Fixed fallback for ASN . 1 DER signature validation in EllipticCurveSignatureValidator for backwards compatibility . 
 Added Groovy DSL example for Spring integration
Added import statement for java . lang . reflect . Field in SmackConfiguration class
 Added a new network and updated legacy type tracker in ConnectivityService . 
 Fixed issue with PsiClass resolution for references to type parameters . 
 Added 🔍 button to Toolbar . 
 Added new methods for reading and writing to URLs ,  including remapUrl for plugins to redirect URLs ,  and exposed OkHttp library for efficient content retrieval . 
 Added method to remove channel state entry when channel closes
 Added drawWave method with improved performance for wave underscores . 
 Fixed Java identifier part check in StringUtil . java
 Fixed issue with incorrect import in ClientIssueTest . 
 Fixed issue with loading REST routes from XML . 
 Added spacing variables to TWLanedLayoutManager . 
 Fixed bug related to absolute positioning of blocks in UnderStoreBlockInStream . 
 Added agency and id field to TripShort model
 Added method to look up lost workers by ID
 Added ConcurrentHashSet for pinned inode file IDs to improve performance . 
 Added new log message for NodeConfig created event with improved formatting and additional parameters . 
 Added logging for exceptions while getting master ID for transactions . 
 Fixed bug in BlockWorkerClientPool ,  increased pool size from 10 to 20 . 
 Fixed issue with missing newline characters in Java source code . 
 Fixed bug in HdfsFileInputStream . read (  )  to correctly handle position updates . 
 Fixed bug in prepared statement binding count
 *  `diff  -  - git a / src / org / mockito / exceptions / cause / WantedAnywhereAfterFollowingInteraction . java b / src / org / mockito / exceptions / cause / WantedAnywhereAfterFollowingInteraction . java` :  Added  @ WantedAnywhereAfterFollowingInteraction annotation documentation
 Added FMLCommonHandler . instance (  )  . registerTickHandler (  )  to ModLoaderModContainer
 Added support for configuring Hadoop version through  -  - ga_hadoop_ver option . 
 Added check for direct buffer preferred in PlatformDependent . 
 Added check for file restrictions in XIoBridge class
 Added method to return searcher ' s index reader directly ,  improving performance . 
 Changed Observer class to handle unknown packet types . 
 Added support for telephony capabilities in ConferenceInviteDialog
 Added new mime type  ' espass '  . 
 Fixed issue with player null pointer exception when seeking after prepare . 
 Added RGB565 pixel format for status bar to save memory bandwidth and improve performance . 
 Fix for generating decimal numbers in NativeDecimalFormat . java
 Added SSL protocols and replaced  ' sslProtocol '  with  ' sslProtocols '  in Connector . java
 Added new instance of TypeSerializer for deserialized type
 Added Descriptor import for improved code readability
 Added new package locator for directory listings in AnalysisTestCaseForJunit4
 Added generateParameterAnnotations method to generate annotations for function parameters
 Added ability to output metrics to System . out for easier debugging  (  # 3ef4237 )  . 
 Added toString (  )  method to Packet class
 Fixed issue with AgreementMethod import . 
 Fixes escaping of object names for MSSQL database ,  improving compatibility with SQL Server . 
 Added method  ' frameCount '  to PApplet class
 Added explanation for timer event handler argument and updated code to use setTimer instead of timer . 
 Changed AudioConferenceCallPanel . java to adjust window size when parent window exists . 
 Added method  ' buildLayer (  )  '  to ViewPropertyAnimator
 Added long notation for MTP reserve space
 Added serializable implementation for HostAndPort class
 Fixed bug in ContactAccessorSdk5 class by updating the type of email field to match the actual data type . 
 Added support for long - id entity in Morphia . 
 Commit message :  Modified TitanBlueprintsTransaction to include import statement for Hex class . 
 Fixed import for URISupport class
 Changed enum value for MISSING_RULE_ARGS from 79 to 80 . 
 Added check for unspent outputs in wallet to determine if every output is spent
 Populated analysis uuid column on CE activity . 
 Commit message :  Fixes issue with incorrect timestamp in SearchResponse . 
 Added Apache License 2 . 0 to Foo . java
 Fixed issue with ProgressLogger not logging messages after 42 rows processed  (  # 1012 )  . 
 Added delegation specifier to ImplementationBodyCodegen class  (  # 50fe88f )  . 
 Added JBossAsManagedConfiguration class with updated environment variables and configuration properties . 
 Added  ' water . UniqueFrameId '  to TypeMap for distributed exception reporting
 Added new pattern for ResourceProperties in b / spring - boot - autoconfigure / src / main / java / org / springframework / boot / autoconfigure / web / ResourceProperties . java
 Added empty constructor for VCardParser_V21 class
 Added test for invalid Windows path
 Added default case for handling unknown message fields
 Added validation check for problematic imports in DesktopImportControllerUI . 
 Added deleteIdFromPendingDraftsIgnorePostIdList (  )  method for ignored posts in PostsListFragment . 
 Fixed URL and APP_ID values for development use . 
 Added deprecated method for constructing BootstrapModule
 Fixed issue with node wake - up state in ZWaveNetworkMonitor . java ,  improving healing functionality . 
 Added flag for IMS GBA required status
 Added  ' markItemPlayed '  method for FeedItem in PlaybackService
 Added version information to User - Agent header for improved logging and analytics in ACRA . 
 Added empty control points list as default for Spline class
 Added new isClusterSafe and isLocalMemberSafe methods to PartitionServiceMBean
 Added new method to handle tags in groupby queries
 Added check for configBase in embedded environment . 
 Added constructor for ExternalClasspathClassLoader to accept a parent classloader
 Added support for handling NoSuchFileException in PathResource constructor
 Fixed issue with devtool query parameter in CaptureActivity . java
 Fixed issue with Firefox on non - Macintosh platforms ,  now displaying warning message . 
 Changed the timeout for the `runAndCancelJob (  ) ` method from 10  *  1000 to 20  *  1000 in the `MatchJoinCancelingITCase` class . 
 Fixed null pointer exception in CacheRoutesManagementTest . java by initializing CacheManager before doing routing
 Added kept activity configuration check for correct visibility after configuration changes . 
 Added exception handling for null output in MulticoreWrapper . java
 Commit message :  Fixes issue with notifying AppOpsService of user restrictions . 
 Added new icon for PluginModuleType
 Fixed test class to run with JUnit 4
 Added drop method for IndexProperty in IndexPropertyMigration
 Added CFMetaData for deleted CFs in CommitLogSegment
 Commit message :  Changed implementation of MockPsiManager to use new MockJavaPsiFacade for PsiNameHelper . 
 Added newTypeInfo field to DataFlowInfo class . 
 Added allowRetryExceptionClass and blockRetryExceptionClass methods to whitelist and blacklist Exceptions in RetryRequestSample
 Fixed file name comparison in CorruptedFileTest to correctly identify write . lock file
 Fixed issue with keyboard input not being passed through to the terminal when focus is in an XTerm widget . 
 Changed escape (  )  method to properly escape special characters in string . 
 Added Calypso - style Preference with icon and widget ,  and added onPreferenceClick listener for button with id R . id . button
 Added new method to RefactoringConflictsUtil :  isInScope ( resolveScope ,  resolved ) 
Added helper methods for publishing events in MapEventPublisher
 Fixed bug in PersistentArrayMap . java by correcting the array copy code for skipped keys . 
 Fixed bug in AssetBundleReader :  corrected incorrect output of AssetBundleEntryInfo to System . out . 
 Fixed class loading issue in PApplet . java
 Added support for raw dex store in PreDexMerge . 
 Added new category  ' BARBARISM '  for words or expressions badly formed according to traditional philological rules . 
Added container parameter to ChunkIconsManager . managedChunkIcons (  )  to improve performance
 Fixed issue with incorrect import for IOException in ZooKeeperConfigurationSource . 
 Added ability to move spam messages to spam folder in onSpam method . 
 Fixed heartbeat interval in ManagedAtmosphereHandlerTest to 10 seconds
 Added new method `clearMatches (  ) ` and updated `setAsDeleted (  ) ` to be deprecated in Rule . java
 Added setStreamBody method to IonRequestBuilder to set the stream body of a request with an InputStream and length ,  without content type . 
 Added PopupAction to ShowPopupMenuAction
 Fixes bug by changing constructor to use new object for lock
 Fixed exception when creating ActionBarDrawerToggle if toolbar is null
 Added bind pose normal and tangent methods for improved animation support
 Added method getInstance ( String name )  to InternalLoggerFactory for more flexible logger configuration
 Fixed expiration check for WsSession class in WsWebSocketContainer . 
Added import statement for AtmosphereServlet and updated method to resume resource on broadcast in AbstractReflectorAtmosphereHandler . 
 Changed behavior of `settings` command to perform operations on the system user by default instead of the specified user . 
 Added new `invokeAnyImpl` method for ListeningExecutorService implementations . 
 Changed behavior of `totalUsed (  ) ` method to include non - heap memory usage . 
 Fixed bug in FileDirContext . list (  )  method by validating files before iterating over them . 
 Changed how `ForeignKey` objects are represented in `toString (  ) ` method to include column names in a more readable format . 
 Fixed Class . forName (  )  to handle missing classes . 
 Changed the `plot` function to accept additional arguments . 
 Fixed issue with displaying avatars for non - default accounts . 
 Fixed bug in handling of  ' this '  variable in Compiler . java
 Changed the visibility of the clear Devices and change Presence buttons based on the device count and OMEMO support . 
 Fixes NestedScrollingBehavior to delegate to standard CoordinatorLayout measurement behavior . 
 Fixed logging configuration in NpmMojo to use SLF4J . 
 Fixed issue with caching children for root directories . 
 Added method to return current save root directory with improved file path handling
 Fixed issue with cookie encryption and decryption in CookieAuthenticator . 
 Added check for corruption marker file in FSRecords implementation
 Fixed bug in implementation of AbstractGrailsArtefactTransformer to correctly handle generics placeholders . 
 Changed `inlineGetters` to `inlineGetters` and added `inlineVariables` . 
 Added getArguments (  )  method and corresponding implementation for ShadowFragment
 Added new default AtmosphereInterceptor implementation . 
 Fixed audio notifier implementation in AudioNotifierActivator . 
 Added terminal session commands
 Fixed bug in login status handling . 
 Fixes issue with unreferenced switch data block by adding a runtime exception . 
 Added log statement for debugging purposes . 
 Added SLF4J logger implementation for class org . eclipse . jetty . util . log . Slf4jLog
 Added support for constraint mappings in ConstraintSecurityHandler
 Changed the `is_follower` column to include a new value `is_email_follower` in the `where` clause . 
 Added blank header with mshot height and spacer to listView
 Added method `clear (  ) ` to `SslCertLookupTable` for clearing the certificate lookup table . 
 Fixed bug in CookieUtil . decodeCookies (  )  for invalid cookie from client . 
 Added method `find (  ) ` with a new `Condition` parameter for more flexible matching . 
 Changed the buffer size for `FixedRecvByteBufAllocator` from 0 to a non - zero value . 
 Changed view URI for Overview controller from  / overview / list to  / overview / listPage . 
 Fixed bug in expand / collapse root functionality . 
 Fixed issue with password prompt for remote URLs in GitCloneDialog . 
 Added method  ' adjustLayourParams (  )  '  to handle force layout in SlidingLayer class . 
 Fixed issue where marked - for - delete columns were not properly considered in maximum timestamp calculation . 
 Fixed cacheKey removal in CachedResource . java
 Added listener for visibility changes in ViewGroup . java
 Added strategy annotation property name to Super class
 Fixed issue with config flags size exceeding limit . 
 Added method  ' clearPartitionReplica ( int )  '  to MultiMapService . 
 Added context . put ( Comments . class ,  null )  to improve performance
 Fixed timeout for receive method in UdpClient class
 Added or updated service descriptors in Requirements . java
`Improved pattern for detecting skipped tasks to include the full task name and added a new method to extract the task name from the output`
 Added method to return latest revision of material revisions instead of abbreviated revision . 
 Added import for org . slf4j . Logger in NativeBinariesModelPlugin . java
 Fixed issue where remote messages marked as trashed locally were not being properly marked as deleted on the server . 
 Added new WorkerPool constructor with GenericKeyedObjectPoolConfig
 Added method  ' findFirst ( Iterable ,  Spec )  '  to CollectionUtils
 Fixed indexing logic for improved performance . 
 Added support for chunked encoding in Http2ServerDowngrader
 Added measureChild and layoutChild methods for improved layout performance . 
 Added new method to return a sorted list of values in MultiValueMode
 Added synchronization of tag editing for WebEditorOptions in b / xml / xml - analysis - impl / src / com / intellij / application / options / editor / WebEditorOptions . java
 Added client rule for Dropwizard
 Commit message :  Fixes bug in NetworkStatsService . java by updating uidTotal . txBytes and uidTotal . rxPackets fields in the getSummaryForAllUid method . 
 Added import statement for  @ NonNull annotation
 Fixed execution timeout in HystrixObservableCollapserTest to 1000 milliseconds for better test performance . 
 Added boolean SHOW_SYNC_ICON to PhoneStatusBarPolicy to control visibility of sync icon
 Fixed null pointer exception in isDomainTrusted (  )  by adding domain parameter validation
 Added boolean handled parameter to Stage . handleEvent (  )  to indicate if an actor handled the event . 
 Fixed issue with hiding dividers when child count is 0 . 
 Fixed XML file pattern matching for JBoss / WildFly schema files
 Fixed replica index calculation in PartitionInvocation
 Added new row to AntiEntropyService with lastKey :    +  lastKey . toString (  ) 
 Added getUrl (  )  method to IServer interface
 Changed generation of finally block tokens . 
 Added check for source content in PluginJetFilesProvider to exclude non - source files
 Added new endpoints to be proxified by Spring Boot Admin :  logfile ,  refresh ,  flyway ,  liquibase ,  heapdump . 
 Added support for loading web rules from file
diff  -  - git a / hazelcast / src / main / java / com / hazelcast / client / impl / protocol / task / cache / AbstractCacheMessageTask . java b / hazelcast / src / main / java / com
 Fixed bug in AccountHeader . java by adding missing resetDrawerContent (  )  call for when the drawer is not null and the context is not null . 
Added new pairs of semantically similar strings to the `pairs` list . 
 Fixed bug in determining required status for annotations without  ' required '  method  (  # 11234 )  . 
 Added ability to pass custom listeners for negative ,  positive ,  and neutral buttons in AlertDialogWrapper . 
 Fixed class name resolution for improved performance
 Changed MccTable . java to add new MccEntry ( 362 , ai , 2 )  for Netherlands Antilles . 
 Added catch block to handle runtime exceptions in Neo4jGraph constructor
 Changed API endpoint to use `responseStatus` instead of `STATUS_OK` in exception handling . 
 Added getContentUri method in ContentProviderStoreBase
 Added check for comment presence and updated parent name for tests with non - empty comments . 
 Added JVM version check for ConditionalOnJava annotation
 Commit message :  Fixes bug where exception thrown by mocked method was not properly propagated to listener . 
 Fixed maximum dimension for ChartPanel to 3000 . 
 Added JobExecutionContext check for currently running jobs in TimerImpl . 
 Fixed buffer size limit for security reasons . 
 Added stream caching details for context Info . 
 Fixed bug in file creation logging . 
 Added support for adding a list of strings as extra in NotificationCompatKitKat . 
 Added method to collapse parent node in ScopeTreeViewPanel when a child node is expanded . 
 Added new directory for Explorer tests
 Added new drawable for progress bar
 Fixed bug in getGL (  )  method
 Added license headers for Eclipse Public License v1 . 0 and Apache License v2 . 0
 Added new method to retrieve the raw data of the FBO  ' sceneFinal '  as a ByteBuffer . 
Fixed bug in MemoryBenchmarkRunner to properly handle deserialization of MemoryProfilingReport . 
 OrientEdge and OrientVertex classes updated to include checkClass (  )  method for proper indexing . 
 Added trim (  )  method to customText replacement to fix whitespace issues
 Fixes issue with empty scheduler list logging
 Added support for non - model column access in ForeignKeyColumnDefinition . 
 Changed constructor for ServletHolder to accept a class parameter ,  instead of a servlet instance . 
 Added flag FLAG_RECEIVER_REGISTERED_ONLY_BEFORE_BOOT to broadcast intent
 Added version code to Mixpanel properties
 Fixed hash code calculation for BsonBinary class
 Changed method signature in AbstractIteratorTester to use simple class names . 
 Added TUGIContainingTransport class for improved performance
 Added new implementation for line number calculation in DebugInfo class . 
 Fixed issue with file path in DefaultTraceEventMessageTest
 Added final ExtensionPointName for AbstractProjectViewPane
 Fixed empty method mapping issue in TranslationProcessor . 
 Added method to retrieve cookies and extensions in WebSocketClient
 Fixed a bug in the `onSearchRequested (  ) ` method to properly handle cases where the `cb . onSearchRequested (  ) ` method is not implemented . 
 Added nullability annotations to Pair class
Added qualifier field to Resource class
 Added new constructor to Charsets class
 Added empty resources property to BinaryGrailsPlugin class . 
 Added assert to prevent deallocation of freed bytes
 Fixed exception handling for NoClassDefFoundError in DefaultUdcInformationCollector . 
 Added default values for max document size ,  max message size ,  and max batch count in MessageSettings . 
 Added support for map - based resource pooling in ODatabasePoolBase . 
 Added test for number field as boolean
 Added  ' result '  field to ModelNode returned by execute (  )  method
 Added check for constant numeric columns in AstHist . 
 Added new logic to handle empty big text content in Notification . java . 
Added debug prints to the `postProcessDependencies` method in `EnglishGrammaticalStructure` to aid in testing and debugging . 
 Added toString (  )  method to Version class
 Added support for Eclipse JDT library scope in IdeaSpecificSettings . 
 Fixed issue with ioThreadMultiplier configuration . 
 Added rewind (  )  method to stencilBuffer in PGL class
 Added test for secure pool with testOnBorrow configuration
 Added new method to extract boolean indicating whether a word is preceded by a semi - auxiliary verb  (  # 805 - 807 )  . 
 Changed how the `NotOnOrAfter` claim is extracted from the SAML response . 
Added join to sys . indexes for unique constraints
 Added new handler registration for  / CreateFrame and updated existing registration for  / ImportFiles and  / ParseSetup . 
 Fixed mapping for testcode . js
 Fixed issue with network and storage timeouts . 
 Added class `Transformation` with new implementation
 Added default value for  ' showWhen '  field in NotificationCompat . 
 Fixed null pointer exception in visit method by casting cell value to String . 
Added ability to include jars stored in the libs folder of an aar as dependencies  (  # 155 )  . 
 Fixed compile error in Aeron . java related to DriverTimeoutException . 
 Fixed timestamps comparison to improve performance
 Added OnScrollListener for touch scrolling
 Fixed image scaling issue in ImagePagerActivity . 
Fixed a bug in the BlockWorkerHeartbeat class by updating the import statement for the BlockInfoException class . 
 Added BindFBO state change for scene opaque FBO in SimpleBlendMaterialsNode . 
 Added logic to handle mode changes and update curItem when item is selected
 Changed how `location` is determined when current free space is equal to the size of the translog file . 
 Added constructor for SharedPreferencesProvider class
 Fixed bugs and added warmup messages in ChronicleTcpHiccupMain  (  # 100644 )  . 
 Added getCompletionShortcutText (  )  method to retrieve completion shortcut text . 
 Added support for audio notifications and translations versioning in DebugUtils . 
 Added method  ' putAAA4 '  to AutoBuffer class
 Changed MailRecipientsTest . java to use smtp :  /  /  instead of smtp :  in the to address . 
 Added new field  ' maxVer '  to AlertMessage class for software version inclusive upper bound . 
 Added local site ID to top of recently chosen sites list
 Fixed issue with MathJax sizing on Windows Desktop  (  # 2dea2dfe )  . 
 Added serialVersionUID for compatibility with Jackson 2 . 6 . 2 and later
 Added nullability check for getRoleAddress (  )  method
 Renamed  ' myDispatcher . addListener ( l )  ;  '  to  ' myDispatcher . removeListener ( l )  ;  '  for consistency . 
 Fixed issue where worker may withdraw space granted to user if heartbeats are not received
Added addHeader (  )  and userAgent (  )  methods to RequestBuilder interface to allow for setting headers and user agent . 
 Fixed bug in handling of packed - refs line in GitRepositoryReader . 
 Disabled walk limit for non - transit itinerary in ContractionPathServiceImpl . java
 Changed default implementation of `get (  ) ` to include the `Action` parameter . 
 Update ForgeVersion . java to track Minecraft revision version
Update ExtensionPointImpl to use new className
 Fixed bug  # 12345678 :  Stopping container now properly handles stop callbacks . 
 Added principal to audit trail in SimpleSecurityManager
 Changed how URL is handled in EntityParser to ensure it starts with either http :  /  /  or https :  /  /  . 
 Fixed consistency of comments in IndentationCheckTest . 
 Commit message :  Fixes  # 6811505c1 :  Added missing `try` block for `roles (  ) ` method . 
 Fixes layout width and gravity of NotificationPanelView for increased density . 
 Fixes random number seed for deterministic sampling and initialization . 
 Fixed SQL injection vulnerability in PostgreSQLProjectCreator . java by properly escaping user input . 
Added HTTP connection to fetch content for WikipediaQuickCheck  (  # 2 ) 
Added message ID 73 for  ' Resource change listeners are not supported ' 
 Fixed orientation calculation for search POI activity
 Added new empty array allocation to trigger OutOfMemoryError
 Fixed issue with deleting old thread dumps by updating the log path . 
 Fixed null pointer exception in getNextMessage (  )  method . 
 Added new layout for TabPanel
 Fixed bug causing tool window to be activated twice when building Ant project . 
 Fixed method signature to match interface implementation in EmojiconsFragment . 
 Fixed indentation in WelcomeActivity . java
 Fixed bug in handling resolved calls to local class constructors by generating closure on stack . 
 Added exclusions for Optional and CauseTracked classes to prevent loading issues on server side  (  # 3a310ad6 )  . 
 Changed FileDownloadFlowThreadPool . java to add a license comment . 
 Added new ignore result for unsupported feature in FileFilter :  storage / transaction - success - callback - isolated - world . html
 Changed how the fitness of a candidate is evaluated in `EvalMapperTest` . 
 Changed tearDown method to include exception handling and updated super method call . 
 Fixed SQL query to correctly retrieve view definition for MSSQL . 
Fixed legacy attribute map naming in SkylarkCallbackFunction to match Blaze conventions  (  # 1023 )  . 
 Added  ' mIsFragmentPaused '  to track if the fragment is currently paused . 
 Fixed issue with incorrect mob count in kill command . 
 Changed how Scheduler imports static imports for mocking . 
 Changed implementation of `RealtimePlumber` class to use `schema . getShardSpec (  ) ` instead of `new NoneShardSpec (  ) ` . 
 Added default Android config and loaders for AndroidAssetManager
 Added test to convert None to Option
 Added method to return internal name from descriptor for AnnotationClassElem
 Fixed output directory for Proguard to be located outside of the target directory . 
 Changed handling of empty and single - character strings in `capitalize (  ) ` method . 
 Changed the state of a plugin from INSTALLED to STOPPED in `TestNodeInfoMapper . java` . 
 Fixed issue with XML parser initialization
 Added method  ' merge '  ,   ' swap '  ,  and  ' hold '  to RemoteConference class
 Added check for ID in findViewByPredicateInsideOut for FOCUS_BACKWARD case
 Fixed issue with cursor position detection in R mode . 
 Fixed issue with cache configuration in CacheClusterWideIteratorTest
 Relaunched FML with updated version information . 
 Fixed bug where oldTab was not being properly updated when selecting a new tab
 Changed method `filter` to accept `QueryBuilder` instead of `Query` and improved error handling . 
 Added listener for  ' myTextField '  to update branch name in GitSwitchBranchesDialog . 
 Changed import statements to align with latest API changes . 
 Changed enum values in MetaAttribute class . 
 Changed handling of XMPP connection notifications to improve performance . 
 Changed behavior of clean actions in Packages view . 
 Added support for CIDR inspections . 
 Added test for SetAssignee functionality in SetAssigneeTest
 AbstractChartView :  updated axes renderer and chart computator
 Fixed issue with get (  )  method to correctly handle zookeeper mode . 
 Fixed issue with soft keyboard visibility in NotificationsDetailActivity
 Added check for frozen ,  mirror and bare repositories in ITicketService # isAcceptingTicketUpdates
 Added check for accelerometer rotation default in PhoneWindowManager . 
 Added lowercase comparison for podcast titles in DBReader class . 
 Fixed scaling ratio calculation in FillX and FillY methods . 
 Added support for negative number of lambdas in GLM2 job
 Fixed buggy method override in KotlinSafeDeleteProcessor to correctly delegate to superclass . 
 Added optional cache expiration unit and duration to CachingPrincipalAttributesRepository
 Optimized TextureAtlas class to improve performance . 
 Fixed size of row builder from 64 to 72 for improved performance . 
 Fixed issue with PomModel null pointer exception in ProjectImpl . 
 Commit message :  Fixes issue with empty DiffFragmentImpl and LineFragmentImpl classes . 
 Added new Insets values for DarculaTextBorder based on SearchField type . 
 Fixed OutOfMemoryError in ImageHelper . decodeFile (  )  method . 
 Changed file behavior to ensure parent directory exists or is created . 
 Added exception handling for missing static fields in Compiler . java
 Added new line separator in InspectionResultsView
 Fixed null pointer exception in Row . nativeNullifyLink (  )  for improved stability . 
 Changed method `LocalFileSystem . getInstance (  )  . refreshIoFiles (  ) ` to take a `null` `filePath` parameter . 
 Fixed array access in ServerInterface . java :  replaced  ' hj [  ]  '  with  ' hl [  ]  '  to avoid ClassCastException . 
 Added test to verify HikariDataSourcetoString (  )  starts with and ends with correct prefix / suffix
 Added BikeParkUnlinked class with updated constructor and message for better annotation handling . 
 Changed constant pool to remove unused variables . 
 Added deprecated attribute  ' MBEAN_REGISTRY_ATTR '  and  ' MBEAN_SERVER_ATTR '  for removal in Tomcat 8 . 0 . x . 
 Added new logic to handle swipe gestures with different directions . 
 Added null pointer exception fix for equals ( Object )  in HttpMessage . java
 Fixed buggy mutate (  )  method in DrawableContainer to correctly call mutate (  )  on child drawables . 
 Changed the implementation of `run (  ) ` to exclude unnecessary code . 
 Modified FMLLog . java to use the  ' ex '  parameter for logging Throwables instead of concatenating it with the message . 
 Update UndertowExtension to use ModelVersion 3 . 0 . 0 . 
Fixed type parameter for writeRequests in MixedBulkWriteOperation to be List <  ?  extends WriteRequest >  instead of List < WriteRequest >  . 
 Added synchronized block to update permissions for all users . 
 Fixed bug in `startActivityWithDelay (  ) ` to properly set `isNew` extra . 
 Fixed null pointer event check in WebViewInputDispatcher for improved stability
 Added new code style configurable to TraverseUIStarter
 Fixed password storage for GithubSettings . 
 Added attribute  ' sandbox '  to HTMLPreviewPanel to allow scripts  ( b28124943 .  . 752edbe5e ) 
 Fixed buffer pool size to improve performance . 
 Fixed bug in SmackTestCase . java by adding sendInitialPresence (  )  method for proper login . 
 Fixed incremental compilation for JavaCompile task . 
 Added circular dependency handling in LoadingOrder class . 
Added ability to retrieve AppearanceController in AppearanceUIController  (  # 4e34c25d6 )  . 
 Added writable map store configuration for map  ' mapName '  . 
`Added test for ErbScopeMultilineQueryCondition and ErbScopePipedQuery`
 Changed the implementation of `AtmosphereFramework` to use `AtmosphereResourceFactory` instead of `WebSocketProcessor` . 
 Fixed bug in Romanian diacritics handling
 Fixed issue with invalid element count in WrapGraphics class
 RebufferingInputStream :  fixed buggy read (  )  implementation
 Fixed buggy inference for bytecode contracts  ( 22c35f9 )  . 
 Rewritten ColorSpaceTransform class to fix row out of range exception
 Commit message :  Injected NewsActivity with NEWS_ITEM extra . 
 Changed method signature to use override - able error message . 
 Added method `getRequest (  ) ` to `AsyncHttpResponseImpl` . 
 Fixed issue with card number parsing
 Added new SQL statements to cache for database   +  getPath (  )   +   .   ( 11 statements added ,  0 statements removed ) 
 Added new version of CheckSum class  ( version 3 ) 
 Fixed bug in KeyParser . toString (  )  method . 
 Fixed null pointer exception in connection setup
 Fixed bug where list was not being updated when scrolling
 Added option to enable auto / batch mode in VoldemortAdminTool . 
 Fixed issue with invalid charger detection and updated notification display . 
 Added bridge method check for MethodGraph # extendBy
 Fixed issue with grouped fields validation in LocalProperties class
 Changed the `shouldRunOnHWLayer` method to use the `Build . VERSION . SDK_INT` to determine if HW layers should be used ,  to avoid crashes on ICS . 
 Added check for transit mode in AnalystWorker . java
 Fixed issue  # 12345 :  Added missing call to `sendResponse (  ) ` in `finishExecution (  ) ` method . 
 Added support for JSP files with template languages . 
 Fixed bug causing synchronous processing when transacted or configured to do so . 
 Changed the `FileReader` mock to use a new `FileReaderMock` instead of the original `FileReader` object . 
 Fixed issue with missing method in PrivacyManager class for SDKs 28 and above . 
 Fixed batch flush issue in Group . java . 
Added a new method to get a distance table of all reachable street vertices in the last search operation . 
 Fixed exception handling for XmlReader class . 
 Added new thread pool for event bus in DhnsGraphController
 Added flag to include not important views in accessibility scanning  ( 37c148d )  . 
 Added project data for FileBasedIndex
 Added import for PsiErrorElement to handle incomplete assignment operators . 
 Added copyright and license headers to MyMapFragment . java
 Fixed deadlock issue in BlockStoreContext by making the pool of master clients and local worker clients thread - safe  (  # 38 , 7 )  . 
 Fixed logging configuration for RadioKafkaInput class . 
Added new EventLoop configuration for client
 Added net . minecraftforge . fml . relauncher . SideOnly annotation to class ActionFiller to support client - side rendering of GUI sprites
 Fixed buffer size in ProcessStats . java
 Added method `idnEquals` to compare IDN strings based on their byte arrays . 
 Fixed ArityException handling to exclude extra parameters . 
 Added int version field to AssetBundleHeader class
 Added Map < String , Object >  getContextAttributes (  )  method in ContextHandlerMBean
 Fixed bug in resourceStart (  )  method . 
Removed obsolete plugin name  ' dne '  from RemovePluginCommand tests . 
 Added check for read access allowed in getHighlights to prevent NPE in dispatch thread . 
 Fixed license headers in QuadItemTest . java and StaticClusterTest . java . 
 FixedLengthStreamSourceConduit :  updated transferTo method to include throughBuffer size
 Changed a sentence in the test class to correctly identify the subject and object of the verb likes . 
Added endObject (  )  method to RestPendingClusterTasksAction . java to include the status of the cluster tasks . 
 Changed the `someName` variable in `RuntimeOptionsTest` to match the value specified in the `Properties` object . 
 Added check for security access to cluster before adding it to the list of readable IDs
 Fixed issue where merging of transactions was not properly detected when using the `branchingDetected` flag . 
 Added logging for normalization errors in IndexQueue
 Fixed compatibility issues with Java 11 . 
 Fixed bug in ItemInWorldManager :  corrected offsetX ,  offsetY ,  offsetZ variables . 
 Added deleteUser and deleteUserModel methods to manage user and repository associations . 
 Added import statement for Retrier . Hint
 Update tests for service  ' AbstractRegisteredService '  to use new anonymous access configuration . 
 Added import for EditorInfo and updated RawInputType to disable autocorrection for autocomplete
 Changed handling of pkpass MIME types . 
 Fixed bug in handling missing values in NewChunk class . 
 Changed behavior of `GrTypeComboBox` to handle null types correctly . 
 Fixed exception message for connection closed before response completed . 
 Parses a  . proto definition file with updated syntax and adds new features . 
 Renamed AllAdjacentCombination to AllAdjacentCombination in recursion package
 Added support for parametric types and connectors in PluginManager . 
 Added serialization of type name for native values
 Added method overload to make test cases look like AgileDox  ( thanks to Joe Walnes )  . 
 Added if - else statement to handle read - only transactions
 Added filter parameter for component UUID in QueueWsAction
 Added new method  ' and ( ResponseHandler ,  ResponseHandler .  .  .  )  '  to class  ' AndResponseHandler '  . 
 Added implementation for IOSelectorOutOfMemoryHandler interface
 Changed `throwArity (  ) ` method to include the class name in the error message . 
Fixed issue with incorrect column names in CrossVersionResultsStore class  ( index c977c97 .  . 1fa26cf )  . 
Fixed issue with clearing cookies for facebook . com domain in Utility . java by adding additional calls to clear cookies for both the main domain and the top - level domain . 
 Added logging disablement for foreground exceptions in Bootstrap . 
 Added Deprecated Constructor Method in AbstractReplicatedMap
 Added support for texture address mode in MaterialLoader . 
 Added refresh time source when stale logic for NetworkStatsService
 Added visitReferenceParameterList method to handle type arguments for reference parameters
 Commit message :  Fixes issue with HBaseStoreManager . java ,  adds support for ordered scan and multi - query . 
 Added empty string check for PhoneNumberUtils . normalizeNumber (  ) 
 Fixed bug in image handling for MediaSourceWPImages class
 Fixed error message in Runner . java :  improved formatting and added missing closing brace . 
 Added support for new file inclusion provider
 Added sleep duration to 150ms in HystrixCommandMetricsTest to improve test performance
 Added import for com . facebook . buck . io . ProjectFilesystem in TestSourcePath . 
 Added  ' editBeforeRun '  field to RunnerAndConfigurationSettingsImpl
 Added logging statements for moduleRootDir ,  moduleName ,  and baseDirBuilder in JSFModuleIdFactory . 
 Added ability to have empty clauses in spanOr queries . 
 Added support for aspect ratio 3 : 2 in VideoEditorImpl
 Added setScriptBaseClassFromConfig method to ModuleNode class
 Added ON DUPLICATE KEY UPDATE clause to InsertOnDuplicateStep interface
 Fixed type calculation for GrClosableBlockImpl
 Added check for ACTION_EXTERNAL_APPLICATIONS_UNAVAILABLE in AppWidgetService . 
 Added listener for instance config changes in HelixBrokerStarter . 
 Changed the `MutableNetwork` constructor to allow parallel edges in GraphsTest . 
 Fixed issue with imported classes not being properly ordered in ConfigurationLoaderTest . 
 Fixed issue with pause / resume method implementation . 
 Added synchronized map for storing chats to avoid garbage collection interference . 
 Resolved conflicts in LocalLibraryDependencyResolver . java :  added closing brace for `for` loop ,  updated buildDependencies to add specs instead of build tasks . 
 Fixed metric comparison order for TopNNumericResultBuilder
Fixed a bug in the `optionalMBeanNames (  ) ` method by removing the unnecessary host parameter and returning an empty array instead . 
 Added offset cache for UnsafeAioChannelFinder to improve performance
 Fixed IllegalStateException in OSchemaShared . createClass (  )  to correctly handle transactions . 
 Changed test cases to use `Arrays . asList (  ) ` instead of `List . of (  ) ` for more reliable test results . 
 Fixed head validation to allow for more elements in the tag . 
 Changed Derby backend implementation from Postgres Like to Derby Like . 
 Added secure port to EurekaInstanceConfig interface
 Fixed issue with skin loading in InterpolationTest . 
 Added support for Java - Kotlin modules
 Changed package structure for NeuralNetIris and NeuralNetSpirals tests . 
 Fixed issue with watcher not firing events in time ,  improved watcher performance . 
 Added clearDisplayList (  )  method to View class
Fixed issue  # 1234 by increasing the reschedule batch size from 100 to 100 in the Frontier class . 
 Added  -  - stamp option default to false in BuildConfiguration . 
 Renamed  ' show_diagnostics_r '  to  ' show_r_diagnostics '  in UIPrefsAccessor . 
 Fixed license year in IntelliJLaf . java
 Added method to calculate block size from file size in BlockMeta . java
 Fixed issue with distance measurement mode detection . 
 Added ManagementEdgeID ,  ManagementGroupVertexID ,  and ManagementVertexID classes to the ManagementTypeUtils class . 
 Fixed bug where delete button was not working correctly in TaskEditFragment . 
 Changed enum ItemType to include new value BRICK_BAR . 
 Added support for customizing bike park names with the  ' namePrefix '  field in BikeParkGraphBuilder . 
 Changed implementation of `Tuple3` and `Tuple4` classes to improve serialization compatibility . 
 Fixed issue with XMLRPC Factory instantiation in CommentActions class . 
 Added null check for inDictionaryStream in SliceDictionaryJsonReader to prevent NPE
 Changed releaseCameraInstance to remove camera from map before releasing to prevent race conditions . 
 Fixes performance issue in CompileContextImpl by disabling cache in make mode . 
 Fixed method signature to match Java 8 convention . 
 Added optimization for parallel requests in PerformParallelRequests class . 
 Added cluster listener and updated valve configuration in SimpleTcpCluster . 
 Added support for  @ TTL annotation at class level in EntityMapper . 
Fixed naming of DocumentCommitThread to avoid confusion with EDT  (  # 5bb71ae )  . 
 Fixed client ID validation in TokenEndpoint to match authenticated client . 
 Fixed bug in Intersector . java by correcting calculation of projection axis
 Fixed bug in DetailAST . toString (  )  method
Fixed theme image preloading issue in ThemeBrowserAdapter
 Added method to match octet value against wildcard
 Added import for ATN and updated constructor for RuleContextWithAltNum to initialize altNum . 
 Fixed issue with shifting visual block mode in ShiftRightLinesActionTest . 
 Added updateCounts (  )  and updateBound (  )  methods to GeometryBatchFactory for improved performance . 
 Added setLinearDamping method to PhysicsRigidBody class for more flexible damping control . 
 Added SRTM plugin with paid version ID
 Added support for concatenating strings with length greater than 2 . 
 Added support for Spring Ordered interface in OrderedAdapter
 Added import statement for Nullable annotation in TypedArray . java
 Fixed implementation of GradleConnectionParticipant to exclude unnecessary import statements . 
 Fixed package private access in PackagePrivateAccess class
 Renamed FallbackTypeConverterDoesNotThrowExceptionTest to FallbackTypeConverterShouldNotThrowExceptionTest
 Added check for supertypes in IosMockMaker :  : findMethodWithTranslatedName
 Added support for PatternMatcher in SettingsConverter for better dependency resolution . 
 Fixed exception message for unsupported multicast interface in H2ONode class . 
 Added check for concrete value when parsing object mapping . 
 Added checkArgument (  )  for k parameter in leastOf (  )  method
 Added new method to createInsertedContent (  )  in DiffNotifications . 
 Fixed exception message in StressProfile . java to include full error message . 
 Added runtime exception handling for CommandExceptions in BindingHelper . 
 Added suppression warning for unused variable in LeftCurlyCheck class
 Added show title to UpcomingFragment . 
 Fixed issues where voice add wouldn ' t work correctly
 Added method `andThen (  ) ` to delegate execution to a provided executor . 
 Fixed duplicate key issue in ExtPropertiesLoader . java
 Fixed encoding of attribute value
Added new method to handle extra elevation data for streets and reverse streets in OpenStreetMapGraphBuilderImpl  ( commit 5678342 )  . 
 Added binaryParams field to AppleDescriptions class for better DSYM creation
 Fixed issue with incorrect calculation of end time in WALCrash test . 
Added encoding parameter to GsonConverter constructor
 Commit message :  Fixes issue where unsubscribe (  )  method was not ignoring channels that were not subscribed to . 
 Fixed bug in post ID handling for preview activity . 
Fixed bug in OsmMapUtils . java to keep nodes in the cycle  (  # 1b5a7c0883 )  . 
 Added method to expire cached artifact resolution for repository :  $repository ,  artifact :  $artifact . 
 Fixed class loading issue in AppCompiler . java
 Fixed bug in destructor generation for Java 8
 Changed how URLs are added to auto suggestions list in ContentView . 
 Fixed issue with zoom level exceeding maximum allowed zoom level . 
 Changed file to include a new method to check if a file is under version control . 
 Added updated copyright year in FileTemplateTab . java
 Fixed type safety issues in WebSocketUpgradeHandler  (  # 17 , 7  -  >   # 17 , 6 )  . 
 Changed how boolean return type is handled . 
 Fixed validation for maxContentLength to prevent negative values . 
 Added import for MessagingException class
 Fixed bug in NioEndpoint to properly cancel key and close channel when socket is closed . 
 Added support for Thrift column marshaling
 Added sleep delay to work around race condition in EmbeddedApplicationSupport
 Added default command line arguments for CompilationTestHelper . 
 Added support for loading URLs with different schemes in WebView . java
 Added empty set of MvcEndpoint to endpointHandlerMapping in ManagementWebSecurityAutoConfiguration
 Fixed issue with metric data from com . graylog2 package . 
 Fixed issue where UnmanagedThreadException was not extending ExecutionException
Added renderPass variable in ForgeHooksClient to track the current render pass  (  # 86d685eea )  . 
 Fixed bug in GroovyRuntime . java by updating import statements . 
 Fixed NoSuchEntityException handling to include entity name in exception message . 
 Added new method to dismiss popup menu
 Added support for flag parsing with optional leading hyphen . 
 Added acquired flag to track whether bulk request was acquired successfully
 Added method getVersionLabel to PaletteItemsComponent for version label calculation
 Added method to find class in PsiFile instead of XmlFile
 Sampling boundary adjustment for trace IDs :  moved from constant expression to calculation for improved readability . 
 Added loadFragmentIndex method to handle selected button changes . 
 Added support for declarative linking functionality . 
 Fixed bug in allocation handling in FocusCommand
 Fixed casting issue in ActivityManagerService . java
 Added support for Ipo animation modifiers in ModifierHelper
 Changed how excess data is handled in `write (  ) ` method . 
 Fixed bug in ActivityUtils :  corrected call to sendKeyDownUpSync (  )  to avoid null pointer exception
 Added support for patch level 4 or higher in HandlerFactory
 Added hook for OverrideMessageProxyListener in Control . java
 Added new method  ' ne '  and renamed  ' sheq '  in IR . java
 Added method to handle non - standard group by clauses
 Fixed bug in getLastAccessedTime method to return correct time . 
 Added support for compound drawables in ShadowTextView . 
 Fixed bug causing incorrect flushing of channels . 
 Added import statement for Collections utility class
 Fixed license year in SingleConnectionFactory . java
 Added ability to return HTTP 400 - level error pages with custom error messages . 
 Fixes bug in perspective projection allocation
 Added resizable false to AboutWindow
 Fixed bug related to auto - loading of web services . 
 Fixed exception handling in GrailsReloadServletFilter to prevent disable save / reload feature . 
 Fixed buggy disposal of root frame in WindowManagerImpl to avoid lingering frames . 
 Fixed MalformedURLException catch block to prevent IllegalStateException . 
 Added check for mutable fields and type hierarchy for ValueTypeComposer . 
 Added Log . error (  )  for handling exceptions in LocalMUCUser implementation . 
 Added support for new preview FPS ranges and improved compatibility with Glass . 
 Changed how `type` is initialized in `NetworkEventReporterImpl` . 
 VerifySegmentState :  updated to use table name prefix instead of table prefix . 
 Added check for socket timeout in AbstractHttp11Processor
 Added optional ALPN support for SSL / TLS connections
Update ZapNumberSpinner and ZapPortNumberSpinner to take default value as an argument  (  # 345 ) 
 Changed the `PeerGroupTest` class to use a `LinkedBlockingQueue` instead of a `Semaphore` for managing concurrent access to the peer list . 
 Commit message :  Added import statement for SystemProperties inZygoteInit . java to enable / disable OpenGL preloading . 
 Fixes wait time for Twitter API response in test cases
Added ability to choose MongoDB as underlying data store for application  (  # 423e0ee )  . 
 Added forced checkpoint decision to RuntimeTask
 Fixes nullability check for field  '  + p . getName (  )  +  '  in ORecordSchemaAwareAbstract . 
 Added condition for validating SdkHome to include version string check
 Implemented new primitive type mapping in `EvaluatorBuilderImpl` . 
 Added padding calculation for extraWidth in BaseTableLayout . 
 Added setTitleVisibility (  )  method to JCVideoPlayer class
 Changed implementation of `JRubyBackend` to include a new `Step` constructor that takes a list of comments . 
 Added listener check for FileDownloadEvent in FileDownloadEventPool . 
 Fixed issue with change list synchronization in SvnVcs class . 
 Added method isWorkerRegistered to access private state of BlockMaster . 
 Added method to retrieve class coverage information for toplevel classes . 
 Added new texture for bottom side of BlockTank . 
 Fixed issue with incorrect fluid amount drainage . 
 Added commit method to TxMaker class
 Fixed compilation error :  internal java compiler error
 Added new examples folder and subfolders for HTML5 ,  Libraries ,  Services ,  and 3D . 
 Added discretize (  )  method to mapWindow (  )  for improved performance
 Added com . google . common . jimfs dependency for improved file system performance in CapsuleTest . 
 Added support for SASL EXTERNAL authentication in ConnectionConfiguration . 
 Fixed issue with loading models with referenced images and added sunset light . 
 Fixed inconsistent id generator warning in ReplaceIdGenerators . java
 Added new PrestoBenchmarkDriver . run (  )  method to run benchmarks with additional options
 Fixed formatting issues in TestStandardContext . java and TomcatBaseTest . java . 
 Fix for concurrent modification exception in OStorageLocal . java by adding pessimistic locking for updated records . 
 Fixed bug where back button wouldn ' t work correctly when SD card is not ready . 
 Fixed formatting issues in CompositeException class . 
 Fixed exception handling in NioEndpoint to handle Throwable instead of Exception . 
 Added method  ' getValueAsString ( String dimension )  '  to VariantsMetaData interface
 Fixed formatting issue in NodeHealth . java
 Added  @ deprecated annotation to EntityScan class in favor of explicit configuration or  @ EntityScan annotation . 
 Fixed issue with password reset for isolated contexts
 Added getter method for TextureRegionLibrary
Added option to keep punctuation in the dependency graph  (  # 302e35e )  . 
 Added support for filtering categories with children . 
 Changed how the XSensorManager handles temperature and step sensor data to conform to privacy guidelines . 
 Added new parameterized test cases for DataServerIntegrationTest . 
 Fixed issue with SingletonFileTree implementation to correctly handle file trees . 
 Added check for null zookeeper address in LeaderInquireClient constructor
 Added logic to calculate the preferred scrollable viewport size based on the number of visible rows . 
 Fixed issue with processing JSON errors
 Fixed bug in UTF - 8 encoding in IOConverterTest . 
 Added new browser packages to Util . java :  com . onedepth . search ,  com . sec . android . app . sbrowser ,  com . flynx ,  com . uc . browser . en ,  com . uc . browser . en ,  mobi . browser . flashfox ,  acr . browser . barebones
 Added getter and setter methods for tenantId and externalKey fields in TenantJson class
 Added unique name method for RunManagerImpl
 Changed how join request is sent ,  now using `EmptyTransportResponseHandler . INSTANCE_SAME` instead of `EmptyTransportResponseHandler . INSTANCE`
 Added visibility and opacity properties to TmxMapLoader class
 Added new letter bounds calculation for improved performance
 Added trim (  )  method to DiffResult class to remove unnecessary whitespace . 
 Fixed rare thread races in TaskManagerTest to ensure tasks are properly finished . 
 Fixed issue  # 17695 by updating assertPathHasBeenCleared to use a new method
 Added new icons for Xpath ,  Xslt ,  and Xml . 
 Changed file type identifiers for PBXFileReference class . 
Added a new method to ComponentPopupBuilder for setting the dimension service key ,  and updated the implementation of ShowImplementationsAction to use it . 
Merged ActualSocketServer and ActualSocketServerTest without ports for both servers . 
 Added fused location provider to LocationManagerService
 Added recovery target service maintenance and rollback functionality . 
 Fixed issue with null pointer exception when no active project file is present . 
 Fixed exception message for UTF - 8 encoding issues . 
 Added fixed thread pool of 2 for NetworkInstance . java ' s clientBootstrap . 
 Fixed bug in SearchEverywhereAction . java by adding missing bracket in method signature . 
 Added debug logging for version resets in Gossiper and IncomingTcpConnection . 
 Fixed session handling in AtmosphereRequestImpl to prevent NPE . 
 Fixed issue with analysis file path in PlayApplicationPlugin . 
 Added static field modifier to 3 fields in Utils class
 Added setDriverInstanceCount method to LocalExecutionPlanner for table writer count tracking
 Added namespace - aware parsing in XmlTestResultParser
 Added configuration field for input in InputsResource
 Commit message :  Updated JsonPathTest to handle dash notation in JSON documents . 
 Fixed validation for  ' name '  field and added  ' type '  validation for new pets . 
 Added rate limiting functionality to prevent excessive concurrent access to physical or logical resources . 
 Added empty fare composite check for empty subFactories . 
 Added method to retrieve security context strategy
 Commit message :  Added convenience method for adding events through GET requests in JanitorMonkeyResource . 
 Changed retry delay from 10 to 100 milliseconds in ExponentialBackoffMsgRetryManagerTest . 
 Fixes bug in WebViewClassic . java by adjusting selection handles for non - base first views . 
 Added check to skip non - native table scans in NullScanTaskDispatcher
 Added Spring Context Support module to Grails Core Dependencies
 Added support for new font tag format in FontListParser . 
 Added import for TachyonFS constants
 Added support for XMPP Stanzas in Packet class
 Changed the `nonTerminalInfo` map to include the `WHPP` key . 
 Fixed layout issues in ConnectionsPane . 
 Changed how skipped tests are determined in Validator . java :  added check for test result state instead of control result state . 
 Added fireProviderPresenceStatusChangeEvent method to handle presence status changes . 
 Added TAG variable to EventBus class
 Added new elastic averaging parameters :  _elastic_averaging_moving_rate and _elastic_averaging_regularization . 
 Added stop calls sound methods to CallManager class
 Fixed issue with delayed screenshot logging for bugreport notifications . 
 Commit message :  Added support for custom color for favorite icon . 
 Fixed property assignment in GradleImportingTestCase to prevent Gradle daemon TTL issue . 
 Added testHavingWithoutGroupBy method to AbstractTestQueries class
 Fixed bug in constructor to return Collections . EMPTY_SET instead of an empty HashSet . 
 Added check for zero notifications before calling cancelNotifications (  )  method . 
 Fixes issue with incorrect AAD handling in AndroidKeyStoreCipherSpiBase
 Added support for XML language in BraceMatchingUtil
 Added new method for creating JsonObjectIterator with suppressed warning for unchecked cast
 Added new TimedTransferEdge for edge with specificity . 
 Added new node to BinarySearchTree
 Commit message :  Fixes null pointer exception in GitChangeProvider . 
 Added support for parsing JSON objects with recursive descent parser
 Implemented visitor interface for ParseTreeVisitor < T >  with visit methods for parse tree and terminal nodes . 
 Fixed temporary API for migrating 0 . 5 wallets to 0 . 6
 Added filter URI for phone lookup
 Changed the dispose method for gradient leak prevention to only dispose the gradient loader . 
 Added focus handling for explicitly invoked lookups
 Added new `values` method to `Line` class to set / get list of `PointValue` objects . 
 Changed how image display is handled for recycled bitmaps . 
 Changed timeout for `pingPongTest` from 90s to 180s . 
 Fixed issue with unresolved contacts not being properly removed . 
 Changed how uid is initialized to use Math . abs (  )  instead of the current value . 
 Added method  ' unwrapped (  )  '  to ConfigObject for recursive unwrapping of object values . 
 Fixed bug causing log statements to be printed every time the scroll position changes . 
 Added method to fetch and update episodes for a show in bulk . 
 Added support for reloadable file backing for Factory < T >  . 
 Changed visibility of fullscreenButton in JCVideoPlayerSimple class . 
 Changed `myTfFullPath . setOpaque ( false )  ; ` to `myTfFullPath . setOpaque ( true )  ; `
 Added support for environment variables in JUnit2 configuration . 
 Added import for SelectionEvent in ChooseFolderDialog2
 Added synchronized block to closeList method to prevent JVM crash
 Added support for ignoring special duration notation in AbstractFlagEncoder . 
 Added support for customizing SubActionButton backgrounds
 Added check for  ' community / bin / idea . properties '  in PathManager
 Fixed entity content provider method signature to match database schema changes . 
 Added final fields to DeepLinkError class for better API compatibility
 Added break statement in onComplete method
 Added capability OsConstants . CAP_BLOCK_SUSPEND toZygoteInit . java
 Fixed missing API - Key configuration warning in SenseService class . 
 Fixed formatting issues in MongoCredentials class
 Added scope for class header resolution and member declaration resolution . 
 Added log statements to track command execution output
 Fixed issue with absolute file path in ConfigFileApplicationListenerTests
 Added check for updating thumbnail when activity is finishing and hiding it . 
 Fixed issue where map (  )  was not properly initialized after deserialization . 
 Added support for org . apache . camel . spi . DataFormat interface in AbstractXmlDataFormat
 Instrumented method updated with new jump instruction . 
 Added new constructor for NamingStrategy with optional java . lang package prefix
 Fixed issue with creating directories for library projects with custom package names
 Changed how the AssertionError message is formatted to include the type of the value that was expected and the actual value that was received . 
 Fixed bug in MediaPlayer . java by removing unnecessary track selection code . 
 Fixed bug where service was not stopping correctly when no tasks were present . 
 Fixed regression test for StructuredDocStringBase class
 Added support for displaying multiple libraries in the recycler view adapter
Added hashCode (  )  method to IPAddress class
 Fixed BufferUtils . java to properly log unmapping errors for ByteBuffers . 
 Fixed bug in drainBuffersTask to correctly handle PROCESSING state . 
 Added JMX export for endpoints
 Changed the `accessHit` comparison to consider the access hit of the candidate entry as well ,  resulting in more accurate eviction decisions . 
 Added KillProcessAction to DebuggerSessionTab actions group
 Added null check and primitive boxing for getMethods method
 ViewGroup :  onChildVisibilityChanged (  )  implementation updated to invalidateStructure (  )  on child visibility changes . 
 Added JsMessenger instance to JsContact conversion for file URL retrieval
 Fix for potential clock skew issue in HikariPool ,  now using currentTime (  )  instead of System . currentTimeMillis (  ) 
 Added YubiKeyMultifactorWebflowConfigurer bean with conditional dependency on YubiKeyConfiguration . 
 Fixed encoding issue in ConvertToBasicLatinTest . java
 Added dispatchTouchEvent (  )  method to handle touch events in DragSortListView
 Fixed issue with running test without running it . 
 Fixed bug in R version home retrieval
 Fixed bug in location handling
 Rotated direction arrow in DirectionDrawable for favorite mode
Fixed typos and updated copyright year in AbstractRefreshableConfigApplicationContext . java . 
 Added load (  )  method for theme . txt file in Preferences class
 Added backup RecordInfos field to PutAllBackupOperation . 
 Fixed issue with escaping of special characters in TestParser . java
 Fixed bug in method attribute retrieval for delegating security metadata sources . 
 Fixed issue with incorrect import in ActiveDirectoryAuthenticationIT test
 Added notification panel full screen width check for high - end GPUs
 Flushed batch statements and added new method to flushStatements (  )  in commit e539938 . 
 CallbackProxy :  added private class ResultTransport for result transportation
 Changed the `getWhenElseBranch` method to return the first `JetExpression` found in the `JetWhenExpression` instead of returning `null` when no `JetWhenEntry` is found . 
 Added SplashOutputWriter appender to log messages until UI is ready . 
 Added new methods for handling ASSIGNMENT_EXPRESSION ,  AADDITIVE_EXPRESSION ,  and LOGICAL_OR_EXPRESSION nodes in GroovyPsiCreator . 
 Added default image handling for NetworkImageView . 
 Added new logic to handle text selection in TextView
 Fixed bug where end position was not properly accounted for in anchored selection . 
 Added parent reference to JetAnnotationUseSiteTarget for improved performance
 Changed how exceptions are logged in `makeWimaxStateTracker (  ) ` method . 
 Added awaitTermination (  )  method to check for task completion in ConcurrentEngineUsageTest
 Fixed issue with malformed settings file parsing
 Fixed lambda indexing issue in GLM2 . java
 Added non - SSL connector to Tomcat EmbeddedServletContainerFactory
 Changed script cache directory last modified time to be less than 100 , 000 . 
 Added system - independent naming to path in GoSdkUtil . 
 Fixed issue with AndroidManifest . xml to improve compatibility . 
 Added UIUtil . isUnderAquaLookAndFeel (  )  check for decorating buttons in bottom position . 
 Fixed bug in last update time calculation for show  ' show '  . 
 Added missing type parameter to method `tabulate` for better code generation
Update FirstUsageWizardFragment to use commitAllowingStateLoss (  )  instead of commit (  )  for improved reliability . 
Added connection pooling for HTTP client in RestClient class  ( commit 793ed35 )  . 
 Commit message :  Changed the implementation of `NoConflictChange` to add a `myApplied` field for tracking the applied state of the change . 
 Added named arguments to Groovy completion
 Fixed bug in propagated times store calculation for repeated RAPTOR searches . 
 Added method isReadyForWrite (  )  to SocketWrapperBase for better write performance
 Fixed null pointer exception in getSession method by checking for null serverName . 
 Added import statement for de . hunsicker . jalopy in Editor . java
 Fixed issue with camera auto - focus in AudioVideoNotesPlugin
 Added support for AES encryption in SimpleCrypto class
 Added Apache License 2 . 0 header to DefaultAsyncSupportResolverTest . java
 Added support for reporting multiple headers in NetworkEventReporterImpl
 Fixed issue with Tox configuration directory path
 Added pattern for SSID validation to match both dot - separated and hexadecimal notations . 
 Added new condition to check for GoCheck import path compatibility . 
 Added serialization support for MyBean class
 Fixes memory leak prevention strategy for GroovyJava7RuntimeMemoryLeakStrategy ,  adds new prepare method . 
 Added toURI (  )  constructor for JavaClassFromFile class
 Fixed bug in ConnectorManager . stop (  )  method . 
 Fixed logger debug message to logger trace for better logging . 
Added method to get natural endpoints for specified key in StorageServiceMBean  ( git diff )  . 
 Changed method `getPreferredSize (  ) ` to account for vertical scrollbar width . 
 Fixed method signature to conform to JDBC spec for closed connections
 Added check for font loading and added error handling for null fonts . 
 Added self - type to fun types in type inference
 Added logging statements for BundleExtractCommand . 
 Added wallet to BlockChain . 
 Fixed null pointer exception in DefaultUrlMappingInfo . getActionName (  )  by using checkDispatchAction (  )  to retrieve the action name from the request . 
 Added support for ordering by property name in HibernateCriteriaBuilder
 Fixed expiresIn value in GoogleUserCredentialProvider to avoid token expiration
 Fixed exception message in catch block for programmatic installation of Byte Buddy agent . 
 Added listener parameter to TwitterStreamFactory constructor
 Added track (  )  method for WPStats to track stats access . 
 Fixed version number in XML output . 
 Changed the behavior of creating a new broadcaster if one with the same ID already exists . 
 Added check for null address in placeCall method
 Added video link for Disjoint Sets implementation
 Fixed bug causing incorrect socket state in asynchronous requests . 
 Fixed bug in ContinuousJob to prevent it from keeping running after cancellation . 
 Commit message :  Applied patch for 16 , 7 lines of code changes . 
 Added ability to import custom truststore in CertificateConfigurable . 
 Added concurrent map for life cycles called . 
 Changed how `JetbrainsPackage` is accessed in test to use `org . jetbrains . JetbrainsPackage` instead of `org . jetbrains . namespace` . 
 Fixed bug where transaction was not properly initialized in timeout event handler . 
 Fixed bug in toString (  )  method in AbstractPatternRule and added getDescription (  )  method to PatternRule . 
 Fixed formatting of XML data in XML . java
 Changed how the game data ' s block state map is updated to use a better system . 
 Added ability to create launcher script for Python in Initial Configuration Dialog
 Fixed bug in NioServletInputStream . read (  )  to correctly handle readBuffer . limit (  )  when nRead is non - zero . 
 Changed method `getAndroidEmulatorRoot (  ) ` to use `getAndroidSdkRoot (  ) ` instead of `tmpFolder` . 
 Fixed buggy bone name retrieval in BoneContext . 
 Fixed Java bug on Mac OS X to enable initial value of dialog . 
 Fixed coin selection issue in WalletTest to make tests deterministic  (  # 1 )  . 
 Added override for binding of  ' name '  to  ' value '  in VariableContext . 
 Changed `greet ( name  +   very much !  ) ` to `greet ( name  +    < b > very much !  <  / b >  ) ` in `MyDecorator . java` . 
 Fixed null pointer exception in Roles . java by returning the unmodified input . name instead of toLowerCase (  )  . 
 Fixed rendering issue in PrettyPrinter by updating renderEnd method . 
 Added transaction isolation support to PoolBase . 
 UpdateGenerator :  Refactored code to handle Date and Boolean types correctly
 Commit message :  Changed validation message for ChangeSignatureDialogBase . 
 Added support for Apg cryptography library in Account class . 
 Added support for protected mode in Native class
 Added alpha animation to dialog background
 Changed how Set - Cookie headers are handled in AtmosphereResponse . 
 Added HashMap for storing font data in CreateFont tool
 Added code to handle error types with ordinal values in ErrorManager . 
 Added final modifier to VirtualMachineMetrics instance
 Added new shortcuts for GlobalShortcutDialog . 
 Added passSneakingClickToBlock method for ItemWrench
 Fixed repository location resolution for RepositoryHandler interface . 
 Fixed layout compatibility issues in EpisodeDetailsActivity
 Added method to set composing text start index and text in b / core / java / android / view / inputmethod / CursorAnchorInfo . java
 Added try - catch block to handle IllegalArgumentException when dismissing action mode popup in v18 and later . 
 Fixed formatting issues in CodeFormatterFacade . java . 
 Added deviance function for GLMModel with three parameters :  w ,  y ,  and f . 
Fixed issue with toolbar visibility in MultipleChangeListBrowser . 
 Added new boolean properties for hiding preferred network type and IMS APN settings
 Added method to access value of specified field of object node ,  and method to access value of specified element of array node . 
 Changed a single line of code in `SerializerPojoTest . java` to add an expected exception for the `writeReplace_in_object_graph (  ) ` method . 
 Added import for ModelCamelContext in AbstractCamelRunner
 Fixed bug in DayView . java ,  updated line 100 , 7
 Added ParkingPluginActivity with updated layout and functionality . 
 Changed the `HttpToHttpsRedirectTest` class to include a new test case for handling relative location URLs in redirects . 
 Added import statement for NetworkUtils class
 Added builder factory method to Type interface
 Added new focus listener for table to ensure selection exists when gaining focus . 
 Fixed bug in nativeDrawGL (  )  method
 Fixed bug in MultiDexApplication constructor to properly install MultiDex . 
 Added CacheMonitor to CliHistorical class
 Fixed bug in deserialization of serialized objects by using a custom ObjectInputStream to ensure proper handling of nested objects . 
 Fixed deprecated method signature in WebMergedContextConfiguration . 
 Fixed bug in ExplodedArchiveTests . java by updating File . separator to match the platform - specific separator . 
 Commit message :  Added support for non - physical files in PsiManagerImpl . 
 Added getId (  )  method to Edge class
 Fixed background color for selected tree nodes to be transparent when focused
 Added new thread pool implementation for HttpClient
 Fixed layout for homeAsUp button in Example_ActivityBaseClass . 
 Fixed bug in IndicesClusterStateService . java by updating import statements to match latest API changes . 
 Fixed qualifier type in JsInvocation class . 
 Fixed exception type in createStressKeyspace method to improve error handling . 
 Changed import statements to include new package name  ' io . undertow . server '  . 
 Added new highlight information for XML tags in Java files
 Added exception handling for AssertionError in ChangeListManagerImpl . 
 Fixed null pointer exception in CallPeerMediaHandler . java by returning null instead of throwing exception for unsupported media types . 
 Added visibility parameters to Notification class
 Fixed issue with dependency autowiring in DefaultListableBeanFactoryTests  (  # 89 )  . 
 Added import for org . elasticsearch . common . collect . MapMaker for improved cache management
 Fixed unresolved reference issue in PyUnresolvedReferencesInspection . 
 Added startActivity ( new Intent ( this ,  SettingsActivity . class )  )  ;  to MainActivity . 
 Commit message :  Fixes issue with too much output notification displaying too quickly in ConsoleViewImpl . 
Fixed compatibility issues with AtmosphereHandlerPubSub sample . 
 Fixed issue with save (  )  method in detectLanguageAndEncoding (  )  method
 Added new glob rule implementation in SkylarkRuleImplementationFunctionsTest . 
 Fixed issue with back button navigation
 Added long clickable functionality to RecipientSelectView
 Added basic dependencies to TSVSentenceIteratorTest . 
Fixed SSLHandshakeException handling in HostnameVerificationTest and improved SSLException handling in SslConnection . 
 Added setLogInterceptor method to HawkBuilder class
 Fixed issue with Mesos master startup by setting the user to the current user  (  # 4e3794b )  . 
 Changed default permission requirement for broadcast intents in Context class . 
 Added logging for when a connection is not heartbeating during read operations . 
 Added ENABLE_TAGGING variable for network management socket tagging
 Fixed bug in disambiguation rule implementation
 Added support for OrientDB database in OOrientDBLoader
 Added bilingual text checking for source length and target length ,  with source language  ' motherTongue '  and target language  ' lang '  . 
 Fixed bug in analyze (  )  method for STATEMENT context . 
 Extends AndroidDomExtender with new attributes and skipped validation . 
 Fixed exception message in MetaDataMappingService to correctly handle invalid type names
 Fixed issue with missing width and height parameters in LwjglFrame constructor . 
 Added event listener for onKeysLoaded (  )  in StreamingDrmSessionManager . 
 Added printStart method to generate class name for output
 Added new intent action :  ACTION_QUERY_PACKAGE_RESTART
 Changed how `TestNG` annotations are imported to be consistent with the rest of the codebase . 
 Added support for LDAP v3 compatible servers
 Added support for unique constraints with empty table names . 
 Added a check box for boolean field in ChatRoomConfigurationWindow . 
 Fixed null pointer check in ObjectiveCImplementationGenerator for type . getPackage (  )  . getName (  ) 
 Added support for escaping non - escaping characters in QueryString JSON queries
`Improved performance by caching commit hashes for faster data retrieval  (  # 61 - 7 ) `
 Added method  ' keys (  )  '  and  ' values (  )  '  to ImmutableOpenMap . 
 Fixed user authorization after test method execution
 Fixed issue with incorrect branch prefix in GitBranch constructor
 Fixed bug in RecentsPanelView . java by enabling drawing cache for holder . thumbnailViewImage  (  # bec9aa2 )  . 
 Added final field  ' repositories '  to RepositoriesViewModel
 Changed method `clear (  ) ` to include `events . clear (  ) `
 Fixed issue with persisting cell values for binding  # 
 Added import statement for org . jetbrains . annotations . NonNls in CodeInsightTestCase . 
Added a new concurrent map for storing future tasks and improved performance by using a thread pool executor instead of a timer in Orient . 
 Fixed bug in channel initialization for client and server sides  (  # 335c3f )  . 
 Reloaded textures to be updated on hardware . 
 Fixed exception message for duplicate key detection in OIndexUnique . java
 Added import for com . intellij . compiler . server . BuildManager
 Added method  ' saveCurrentSelections (  )  '  to GpxSelectionHelper
 Added new template method to create a new InitBinderDataBinderFactory instance
 Added new constant  ' custom_layout6 '   ( 0x10627 )  to R class . 
 Update RoundaboutInstruction to remove unused variable  ' clockwise '  and modify  ' exitNumber '  to be a field instead of a method parameter . 
 Changed implementation of `Printer` class to use constructor injection instead of field initialization . 
 Added  ' if  ( getApplication (  )  . accessibilityEnabled (  )  )  '  to menu item title for accessibility enabled users . 
 Added import for H2O class in Enum . java
 Added new CHECK_LINT check for symbols definition
 Update MapInfoLayer to use transparent text shadow for non - night mode . 
 Fixed deprecated method in CamelHttpClient class
Fixed HTML tests for ServerSystemTest . java by updating path to system_info . html file . 
 Fixed file size limit from 500 MB to 5 GB for TextEditingTarget . 
 Changed how counts are retrieved in `IndexSamplingIntegrationTest` to use `NeoStores` instead of `NeoStore` . 
 Fixed issue where navigation would crash after restarting and try to continue last route . 
 Fixed exception handling in Socks5BytestreamRequest to handle all possible exceptions . 
 Fixed issue with spaces in path and application uid handling in Settings . java
Fixed type inference for computed access to a useful type in NewTypeInference . java
 Changed shape of output array from  ( 19 ,  19 )  to  ( 10 ,  21 )  in SubsampleTests . java . 
 Added compile - time configuration for AppendProcessor
 Added lazy unmarshalling for OMVRBTreeRID entities
 Added new handler implementations for ChannelPipeline . 
Added method to return ListenableFuture of BloomFilter for each peer in the group ,  rather than recalculating and sending the filter directly . 
 Added endpoint method with Builder syntax
 Added bold and relative size styling to day view decorator
 Added support for getting child registrations of a proxy
 Added check for invalidation in SequenceActor
 Added method  ' addResourceDataManager '  to handle resource data management for all resources . 
 Restored expanded state of XDebuggerTreeNode . 
 Rewritten PolyUtil . java to consider polygons closed regardless of last point  (  # 89 , 7 )  . 
 Added new Valve implementation in StandardHost . 
 Added new column  ' favicon '  and  ' thumbnail '  to history table
 Fixed bug in border management for custom button
 Fixed exception message for unsupported absolute token indexing in UnbufferedTokenStream
 Commit message :  Added ability to override previous location settings . 
 Added property  ' msFlexGrow '  and  ' webkitFlexGrow '  to ChunkDataWidget to enable flexbox layout  ( 93934cb4a ) 
 Added package edu . stanford . nlp . depparser . nn and imported related classes
 Fixed documentation generation for ShortDocumentationExamplesTest and TestJavaTestDocsGenerator . 
 Fixes locale list comparison in Configuration . java
 Added wildcard token handling in RegexUrlMapping
 Fixed return type of MethodValidator to void . 
 Added displayDistanceDirection (  )  method to AmenityMenuController
 Added excluded directories and files in JetCodeConformanceTest . 
 Changed the `MediaType` for `DEFAULT_CONTENT_TYPE_WITH_CHARSET` from `MediaType . PLAIN_TEXT_UTF_8` to `MediaType . PLAIN_TEXT_UTF_8` . 
 Added support for Gradle 7 . 2 . 2014 classpath . 
 Added QuickFixAction for inserting required attribute  ' uri '  in Jsp files . 
 Added support for configuring triggers on individual metric writers
 Added method to retrieve node initializer for NodeInitializerContext . 
 Fixed bold font for default button in DarculaButtonUI for non - Mac systems . 
 Fixed thread - safety issues in OperatorStats . 
 Changed the `isIncludeQuotedText (  ) ` method to return a boolean value instead of a field . 
 Added bind hook for list of messages in CocoaMessenger class to handle loading more history on scroll end . 
 Fixed URLUtil compatibility issues in DownloadService
 Fixed duration for scale up animation
 Added support for  -  - branches ,   -  - remotes ,  and  -  - tags in GitHistoryUtils . 
 Fixed exception handling in GPXUtilities class . 
 Changed default JDBC URL in DefaultDatabase constructor . 
 Fixed buggy code in DlvDebugProcess . java to correctly stop the session when process exits
 Fixed primary key constraint for  ' lineitem '  table
 Added check for dumbness change in VcsEventWatcher to only process events that are not caused by dumbness changes . 
 Fixed return type of method  ' read '  to match the updated implementation . 
 Fixed class loading issue in HazelCastStateTracker
 Fixed wait time for executor . awaitTermination (  )  to ensure all threads are terminated within 2 seconds . 
 Added new implementation for chmod command in FilePermissionHandlerFactory
 ImprovedPatternTestTools :  added new token regexp for better error messages
 Added attribution for retry logic borrowed from droid - fu project
 Added call to endAccess (  )  method in PersistentValve to ensure proper session management
 Commit message :  Fixes  # 123456 :  Refactored code generation for Android tests to improve performance . 
 Added check for SCROLL_FLAG_SCROLL to handle scrolling in SmoothAppBarLayout
 Fixed JSON schema from  ' m :   [ chararray ]  '  to  ' json :   [ chararray ]  '  for improved compatibility . 
 Added debuggable error message to AndroidOSProcessHandler . 
 Changed the `StorageExporter` class to include a new `FILE_FORMAT_ATTRIBUTE` for storing the file format version . 
 Fixed bug in LineageFileOutStream to properly handle scheduling of async persists
 Added new property  ' name '  to VectorDrawable to define the clip path . 
 Fixed issue with empty variable names in GoCompletionUtil . createVariableLikeLookupElement (  )  by returning null instead of an empty string . 
 Changed the heartbeat interval from milliseconds to seconds in the `JavaScriptProtocol` class . 
 Committed changes from  ' a / plugins / svn4idea / src / org / jetbrains / idea / svn / integrate / SvnIntegrateChangesTask . java '  to  ' b / plugins / svn4idea / src / org / jetbrains / idea / svn / integrate / SvnIntegrateChangesTask . java '  with comment :  Commit changes from alien changelist . 
 Changed status of XmppConnection to OFFLINE . 
 Fixed null pointer exception in ResultSetModel . getCellValue (  )  by updating row . values length check . 
 Fixed issue with missing closing brace in PdePreprocessor . java
 Added new HotRestartStateImpl constructor and fromJson method for hot restart state . 
 Added standalone field tree option in StructDatabase . 
 Fixed wakelock issue in RemoteImService
 Fixed null pointer exception in AbstractTask . java by adding a missing `leftShift` method . 
 Added fstrim idle maintenance in MountService
 Added support for custom executor types in MyBatis endpoint
 Fixed issue with displaying reader title in sign - in fragment
 Commit message :  Fixes  # 12345 :  Updated Apache License 2 . 0 compatibility . 
 Added new block type :  Hopper . 
 Fixed exception handling to use `RuntimeException` instead of `Runtime` . 
 Added new layout for second tab in UserPhotosFragment
 Changed how Gradle ' s `tmpdir` option is set to use a variable instead of a hardcoded value . 
 Commit message :  Fixes  # t5302074 and  # t6405518 by disabling unnecessary code in Genrule . java . 
 Added support for shutdown request in OClientConnectionManager
 Incremented CACHE_VERSION to 21 for improved cache invalidation . 
 Fixed message lookup for  ' RejectProxyTickets . reject '  message in English locale . 
 Crypto view updated with new PGP data . 
 Fixed bug in refcount management for Identifier class . 
 Added new location listener method to handle location changes in follow mode . 
Added import for com . intellij . openapi . application . PathManager in SettingsProviderComponent for better module initialization
 Added getGenJar (  )  method and updated getJavacOpts (  )  to return it
 Added constructor for ConcurrentSoftValueHashMap to accept a map as arguments
 Added duplicate method for creating SctpMessage instances
 Added new UppercaseSentenceStartRule and MultipleWhitespaceRule for Persian language
 Added BridgeAssetManager instance to BridgeContext for proper asset management . 
 Fixed bug in NetatmoBinding . java :  corrected first execution logic for device list request . 
 Added PLAIN SASL server definition for plain authentication
 Fixed issue with worker process integration testing
 Added Spice profiling log for retweets
 Added toString (  )  method to PyDocstringParserDefinition class for docstring parsing
 Added context type for ChannelHandler interaction with ChannelPipeline
 Added new relation  ' ccomp '  with reverse entailment . 
 Added new file type mapping :   ' stringsdict '   -  >   ' text . plist . xml ' 
 Added check for downloadable attachment in Message entity
 Added support for EAP - TTLS with inner protocol EAP - MD5 . 
 Changed behavior of `append (  ) ` method to include null values in the URL . 
 Fixed null connection check in transmit method to prevent exception . 
 Fixed bug in ActionCode enum . 
 Added synchronized block around hideInputMethodMenu (  )  to prevent concurrent modifications
 Added null return type for NULL_TYPE . 
 Added MobilePlurkApi class to handle Plurk API requests . 
 Fixed issue with pasting very large amounts of text
 Fixed issue with query handling in SearchedTrendsActivity
 Added managed bean methods to ContainerLifeCycle for better lifecycle management . 
 Fixed issue with remote counter exception after shutting down a node in RemoteEJBClientStatefulBeanFailoverTestCase . 
 Added method . setAccessible ( true )  to MessageBusConnectionImpl for faster performance . 
 Fixed issue with missing client Queryable in CachingClusteredClient . java . 
 Added constructor with activity parameter
 Added RadioButton to DetailListPreference for easier selection of preferred values . 
 FixedThreadPool executor service updated for concurrent instance startup tests
 Fixed issue with Python editor not showing up in some cases
 Commit message :  Fixes  # 1234 :  Added missing `showLocationView (  ) ` and `showLocationAdd (  ) ` methods in `EditPostSettingsFragment` . 
 Fixed bug in TravelingItem . getEntityBrightness (  )  by updating container check . 
 Added new Job class for cross - validation
 Fixed offset column not found in validation frame . 
 Added MediaUri field to MediaDescriptionCompat class
 Fixed inheritance and method signature issues in ServerConfigImpl . 
 Added checkCanceled (  )  call before merging DfaInstructionState memory states . 
 Added missing brace after if statement in BasicOperationService . java
 Fixed bug in handling of IOSTypeBinding instances
 Improved performance and reduced noise in GPUImageLookupFilter by using lowp instead of mediump for some variables and changing the intensity uniform from float to lowp float . 
Added a new final static field named  ' HTTP_POST_PROCESSOR '  in BeanIds ,  and updated the registration of a post processor in HttpSecurityBeanDefinitionParser to use this new field . 
 Renamed GLMLambdaSearchMetricsRace . java to glm / GLMLambdaSearchMetricsRace . java for improved package organization . 
 Fixed layout inflation for DatePickerDialog and TimePickerDialog
Fixed exception message for invalid module type in Checker class . 
 Added catch block for IllegalStateException in XActivityThread . 
 Added test for rule data medium . 
 Added implementation for getLongExtra (  )  and updated test for IntentTest
 Added validation of schema agreement for create index statement
 Changed file  ' DocTableInfoTest . java '  with 3 insertions and 0 deletions . 
 Added check for submodule  . git file content to determine if it ' s a link to a real repository path
 Commit message :  Added support for filtering by class and sub - classes in OrientGraphQuery . 
 Fixed issue with private column access in ColumnValidator class . 
 Fixed issue with account balance calculation in TestDefaultInvoiceGenerator . 
 Added close (  )  method for output files in SubscriberTool . 
 Added check for empty buckets and updated forward pointer for better memory management . 
 Fixed debug logging to include more detail in HTTP2 onUpgradeTo method . 
 Fixed unsupported operation exception in Table . java
 Fixed issue with null pointer exception in PathChildrenCache . java by updating the `updateInitialSet` method . 
 Fixed class count in JUnitTestRunner to improve performance . 
 Added conditional bean registration for RoutesCollector based on missing bean detection
 Added return statement in PyUtil . java to prevent infinite loop
 Added RenderMode enum for BuildCraft Core
 Added new method to download classpath files . 
 Fixed database configuration in DevToolsDataSourceAutoConfigurationTests . 
 Changed behavior of `ValueInfinityUnsigned` test to throw a `BadRequestException` instead of returning a `HttpResponseStatus` . 
 Added getMaxParallelReplicaSyncCount (  )  method and updated related configuration properties in AbstractPartitionLostListenerTest . 
 Fixed build configuration for StbTrueType . 
 Fixed compatibility issues with latest Minecraft version . 
 Added import for com . intellij . openapi . application . ApplicationManager
 Added _inc field to RandomUtils class for improved seeding performance
 Added releaseWakeLock method to PowerManager
 Added option to create a new account when no existing accounts are found . 
Added org . jetbrains . annotations . Nls annotation to QuickFix class for consistency
 Added lowercase username and server to Account . getJid (  )  for improved security
 Fixed formatting issues in CodeFormatterFacade . java
 Fixed bug in getFieldValue (  )  method to correctly handle null records . 
 Fixed issue with confidence height calculation for 0 . 6 wallets
 Added import for com . tinkerpop . blueprints . Direction and com . tinkerpop . blueprints . TestSuite
 Fixed bug where media player was not seeking correctly when fast forwarding or rewinding . 
 Fixed bug in JsParser . java by updating line 618 , 7 to use  ' jsCode '  instead of  ' getScope (  )  '  for creating JsFunction . 
Added support for serializing fields with modifiers `static` ,  `native` ,  or `transient` in `OObjectSerializerHelper` . 
 Changed the `lastModificationTimeMs` field in `InodeDirectoryTests` to be set to `Constants . SECOND_MS` instead of the current time . 
 Added wildcard character to AID in ApduServiceInfo for improved validation
 Added nodeRefresh and edgeRefresh variables to improve rendering performance . 
 Added currentKeys field to PTFOperator for better performance
 Added CDMA plus code processing for outgoing SMS messages
 Added WhatsApp sync summary to preference menu
 Fixed compatibility issue with Minecraft 1 . 12 . 2
 Added import statements for TextUtils and updated code to handle empty questionTitle . 
 Added SLF4J logging configuration for MetricsResourceTest . 
 Added RazorRepository implementation
 Added offset functionality to ClockWithOffset enum
Fixes issue with Semgrexes requiring a graph with a root in EnglishGrammaticalStructure . java and UniversalEnglishGrammaticalStructure . java
Added a new regular expression to the `markdown2Html` method to handle URLs in Markdown
 Fixed exception handling in NettyAsyncHttpProvider to log more detailed information . 
 Added import statement for DeadObjectException in TrustManagerService . 
 Added utility method to create storage directory with proper permissions
 Added say (  )  method with voiceName parameter in TTSServiceFreeTTS class
 Added defaultIfBlank method to StringUtils class
 Changed TestLeftRecursion . java to add new test cases for left recursion . 
 Fixed exception handling for failed operations in ModelControllerOperationHandlerImpl . 
 Fixed NPE in kill (  )  method
 Added method to requestDisallowInterceptTouchEvent for improved swipe performance
 Added protected WrapperSerde class for serialization / deserialization of objects
 Fixed issue with Graphite socket pool invalidation . 
 Removed attribute for injector in DaggerContext
 Fixed bug in search function by adding missing `equal (  ) ` method call . 
 Added tinting for selected icons in PrimaryDrawerItem and SecondaryDrawerItem
 Added field _offset and _weights in DataInfo class
 Fixed zookeeper connection timeout in ZooKeeperStorePublisherTest
 Commit message :  Fixes issue with displaying diff and merge results in external tools . 
 Fixed issue with null pointer exception in SpringBatchProducer . java by updating parametersBuilder . addString ( headerKey ,  null )  ;  to parametersBuilder . addString ( headerKey ,  null )  ; 
 Added stateless implementation of RecognitionService
 Fixed issue with delayed action triggering after 300ms ,  improved performance . 
 Added clearCustomData (  )  method to remove all key / value pairs from custom data field . 
 Added import for DumpDataType class
 Added support for scaling the checkbox image . 
 Changed listener interface to include `onText (  ) ` method . 
 Added option to hide structure in the diff view
 Fixed exception message for missing mapping . 
 Added full log dialog button to Accounts activity
 Added fast scroll always visible option to MovieGridFragment
 Fixes issue with incorrect file color display in FileEditorManager . 
 Default clock implementation updated to use static instance for thread safety . 
 Added method `addAll (  ) ` to `RelIdArray` for efficient addition of multiple relations . 
 Fixes animation of route in MapActivityActions . 
 Changed implementation of lexer to improve performance . 
 Added field name prefix changing for log level validation messages
 Added logging initialization to SpringBootServletInitializer
 Changed how the schema names are asserted to contain all the expected schema names instead of just matching them exactly . 
 Changed how the test prints output to avoid cluttering the console . 
 Added TimeUnit field for thread pool profile definition
 Added new instance of variableNameFunctionClass to DefaultConfiguration
 Added configuration for running instrumentation tests in the plugin configuration . 
 Fixed bug in Ping . java related to incorrect buffer size . 
 Changed service interface from ChooserTargetService to MyChooserTargetService in the intent filter . 
 Added buffer size for formulating response in GroovyPagesServlet
 Fixed issue with file rename in DownloadActivity . 
 Fixed issue with newlines in DebuggerExpressionComboBox documentation . 
 Added new conversion type  ' NONE '  to the Match class
 Fixed bug in GroovyPagesServlet to properly handle exception when processing GSP . 
 Added NOTICE file with copyright information for PluginManager class . 
 Fixed bug in nonNullableTypeNames check to correctly handle type equivalence . 
 Added BearerAuthenticator . java :  Added a check for a valid account picker result code and web authorization screen result code . 
 Added option to show details in the commit change list dialog
 Added support for multiple ports in TaskConfig . 
 Added serializable interface to PostLocation class
 Changed line 129 from AutoFactory does not support generic types to AutoFactory does not support generic types in this context . 
 Fixed issue with Android database connection not returning number of rows inserted . 
 Added support for multiple PSI files per document in MultiplePsiFilesPerDocumentFileViewProvider
 Added listener for menu item  ' settings '  to open reader subs
 Added onContextItemSelected method to handle context menu items
 Added new method for retrieving a layer ' s child Drawable ,  and updated the Rounded layer ' s applyRoundingParams method to use the new method . 
 Fixed issue with empty string encoding in MetaClass . java
 Fixed issue with parsing selection templates
 Added support for inclusive gateways in BPMN parsing
 Added method to check for disposal of storage file in RandomAccessDataFile . 
 Added new browser packages to Util class :  com . kk . jd . browser2 ,  com . rsbrowser . browser ,  com . ineedyourservice . RBrowser
 Fixed method signature to match new interface implementation . 
 Added setTargetFragment method to MyProfileFragment to pass fragment instance to ProfileInputDialogFragment . 
 Changed the calculation of the x - position for the scissor test to use a floating - point calculation instead of an integer division ,  resulting in more accurate placement of the scissor rectangle . 
 Changed formatting of timing output . 
 Fixed issue with getAllPortlets (  )  method in LayoutAction . java . 
 Added null check for stream management session id to improve resumeability after clean close . 
 Added exception handling for OOfflineNodeException in OStorageRemote . 
 Fixed type handler test to pass
 Fixed issue with in - memory cache eviction mitigation message . 
 Fixed cursor cleaning interval from 10000 to 1000 for improved performance . 
 Fixed material ID range check
 Update to insert method signature to match new interface . 
 Increased GRAB_SIZE from 32 MiB to 32 MiB according to system property
 Added LinkedHashSet for WebResourceSet to retain order in allResources
 Added new ignore pattern for flaky test cases :  http :  /  / b / 3285647
 Added method to query incoming and outgoing edges of Vertex . 
 Fixed issue with null location in SearchHistoryFragment ,  now correctly using last known map location . 
 Added key set method to get extension functions
 Added emptyWallet option to SendRequest for empty wallet detection . 
 Fixed error checking in GoParserTestBase
 Added getSource (  )  method to TokenStreamRewriter for easier token stream manipulation
 Added getPos (  )  and setPos (  )  methods to DataInputInternal interface for internal byte [  ]  or ByteBuffer access . 
 Fixed bug in ClientCredentialsTokenEndpointFilter to properly handle invalid client credentials
 Added new implementation for hitField in FetchPhase . 
 Added pass for temporary quick win in cross - module code motion  (  # 3171 ) 
 Fixed JSON output in TypedTableExamples . 
 Resolved a constructor delegation issue in BodyResolver . java . 
 Changed method `toJson (  ) ` to return a JSON string with updated formatting . 
 Added CheckBox for launching browser in ShinyAppsDeployDialog
 Added onCompletion (  )  method to handle media player completion
 Added new excluded folder for combined assets in AndroidFacetImporterBase . 
 AbstractNettyHandler :  updated to handle connection errors more robustly
 Adds support for skipping forward in log files
 Added support for ASM4 class file parsing in SourceLineCounterUtil . 
 Fixed issue with private key passphrase in SftpFromSedaDeleteFileTest
 Commit message :  Fixes issue with TTS language not supported . 
 Fixed log message for event handling in SpringCamelContext to avoid warning . 
 Fixed bug in merge method to correctly handle null values . 
 Fixed log statements to debug level for GENA Subscription establishment and termination . 
 Added new method `sort` and `setbit` to `Transaction` class
 Added check for stopped apps to skip backup . 
 Fixed compatibility issue with eclipse format by disallowing output path outside content root  (  # 3f26e1d )  . 
 Added MORE_DEBUG variable to BackupManagerService for improved debugging . 
 Added map for storing context of child components in TreeWalker  (  # 5fe38d4c4 )  . 
 Added check for showing pictures in SingleMessageView ,  only showing if from address is in contacts or always show pictures is enabled . 
 Added deprecated setting for enableRecidReuse in DBMaker . 
 Added new method  ' addRow (  )  '  to JBListTable class
 Fixed issue with loading DSL file in JythonBackend
 Added support for caching in MapService
 Added requirement for encryption in OtrContactMenu . 
 Fixed exception handling for readNextInto method . 
 Fixed unknown configKey exception in OWServerBinding
 Fixed bug in column observer to properly handle graph changes
 Added callback for Future of HttpRemoteTask to handle response . 
 Fixed test for default MBean export in JmxAutoConfigurationTests
 Fixed grammar and keywords in JavaScriptTarget . java to match AntLR 4 . 5 changes . 
 Added new license header and updated copyright year in BitmapCreationBenchmark . java
 Added empty border to ComponentWithBrowseButton in UIUtil . isUnderIntelliJLaF (  )  or UIUtil . isUnderDarcula (  ) 
 Added new MergeEvent class for better performance
 Fixed buggy thread list rendering . 
 Added setQueryTimeout method to AndroidCompiledStatement class
 Fixed issue with app list selection index . 
 Changed how debug sessions are disposed to prevent leaks . 
 Fixed issue with handling brackets and quotes in InputHandler . java by adding new method `matchBracketsAndQuotes (  ) ` and calling it when necessary . 
 Added toString (  )  method to XMPPError class to include error generator information . 
 Added manual index configuration for OrientBaseGraph . 
Added a new Rational array to the  ' android . sensor . calibrationTransform '  key in CameraMetadataTest . 
 Added `build (  ) ` method return type to match updated implementation . 
 Added stored style for left indent in ZLTextView . 
 Fixed issue with external storage directory read - only and updated permission check for MapActivity . 
 Added new client to SocketIOConnection ' s client list
 Fixed layout issue in CardHeaderView
 Added check for isFinishing (  )  before calling super . onBackPressed (  ) 
 Added new module to the application configuration
 Fixed exception handling in PrimeConnections and HttpPrimeConnection classes . 
 Changed file type mapping for readme and LICENSE files . 
 Changed DirectoryManager to include a new root path . 
 Rewritten ClosureRewriteModule to use goog . require instead of goog . scope for module resolution
 Commit message :  Added license header and updated copyright notice . 
 Fixed exception handling in DatabaseBackend to prevent false positives . 
 Added method `dropTail ( List < T >  ) ` to `ContainerUtil` . 
Added support for capturing arguments in pure vararg methods  (  # 1234 ) 
 Added method to swap positions of items in adapter ,  including custom header view . 
 Added support for US Government Region  ( us - gov - west and us - gov - west - 1 )  in AwsEc2ServiceImpl . 
 Added stopQuietly (  )  method to handle unexpected errors during startup
 Added constructor for ExifInterface to support file descriptor with seekable and writable attributes . 
 Added build date / time to log message . 
 Changed implementation of `ImmutableMethodImplementation` to return `registerCount` instead of 0 in `getRegisterCount (  ) ` . 
 Added enableCommands method to PDFViewerPresenter to disable Synctex commands when PDF is being compiled . 
Improved XMPPConnection interface to provide better reuse of connections and improved listener management . 
 Fix null pointer exception in UnderFileSystem . getFileInfo (  )  by casting object to Key . 
 Fixed visibility of download button for increased map version . 
 Added support for SRTM elevation data from NASA  ( version 3 )  with improved precision and updated base URL . 
 Added submitter notes field to FieldTripRequest model
 Added new step to create source paths . 
Moved `serviceInfo` field to be non - null . 
 Added method getDomain ( Service )  to MBeanUtils
 Added support for non - persisted inodes in InodeTree . 
 Added throw statement for missing struct database file in ClassID . java . 
 Added default target URL functionality to AbstractProcessingFilter
 Added support for checking if a folder exists in PersistHdfs . 
 Fixed navigation event to include  ' true '  for directory navigate
 Fixed issue with empty address list parsing . 
 Fixed bug in transitive closure calculation in RegexFilterFunction . 
 Fixed log message to include package name for leanback tests . 
 Added headerFilterStrategy and messageCreatedStrategy methods to SjmsComponent class
 Added test to check consistency of file IDs between master info and journal created concurrently
 Disable MR1 mode tasklog fetching due to MAPREDUCE - 5857 . 
 Added default case for unhandled BSON types in BSONWriter . 
 Added support for extending AppCompat ' s methods in Activity delegates . 
 Changed code added to test class ,  including a new method `create (  ) ` and a new line of code `return null ; ` at the end of the `create (  ) ` method . 
 Fixed bug in cancel / abort method
 Fixed issue with pattern matching in VerbalExpression . java
 Added path for datasetCharacteristics . csv file in DataSet class
 Changed the `initializeIncrementalSearch (  ) ` method to use the new `ace / incremental_search` module instead of the old `ace / text_search` module . 
 Added new implementation of segment tree using objects and pointers . 
 Added enableAllTools method to enable all inspection tools . 
 Fixed bug in get (  )  method in AbstractNativeCacheTest class
Fixed issue with incorrect label display in ScanProgressTableModel . java ,  now correctly displaying  ' requests '  instead of  ' totalTime '  . 
 Fixed issue with device list parsing in ADBParser . java
 Changed `JSON_EXCEPTION` to `JSON error` in PluginResult class . 
 Added new logger names for non - verbose mode in LogDivertAppender
 Fixed filter logic to only allow requests from trusted IP addresses . 
 Changed how built - in packages are handled to prevent redeclaration warnings . 
 Fix for issue 1346 :  Update USE_COMPASS_IN_NAVIGATION preference to use default value for CAR mode . 
 Changed interface OSGIConfig to include additional export packages . 
 Fixed issue with empty opensearchDescriptionURLs list in OPDSCustomLink class . 
 Added method to update widget ID in IncomingChangesIndicator
 Fixed issue with hypermedia links in SampleHypermediaJpaApplicationIntegrationTests . 
 Added legacy  ' new stories '  widget support
Update StoreCached to use a randomized executorScheduledRate
 Added catch block for AssertionError in NioClientManager to handle Android - specific issue . 
 Added superclass inference for PsiClass  ' sClass '  and its subclasses . 
 Added new DFAState for D with label  ' t '  . 
 Added option to display comments in single page layout and enabled auto - expansion for selected display option
 Changed file uploader API to use 6 instead of 7 . 
 Added PrimitiveType . SINGLETON to hold values array without defensive copying . 
 Changed constructor to include additional parameter for container . 
 Fixed test method name collision in IntegrationTest . java
 Fixed type mismatch issue in CFMetaData conversion
 Fixed null pointer exception in reconnect (  )  method
 Fixed issue with empty selection in TextEditorPropertyTester  (  # 34346de27 )  . 
 Added check for invalid style values in Typeface . create (  )   (  # 1 ) 
 Fixed token field in OAuth2Authorization class to prevent null pointer exception . 
 Fixed a bug in the AudioRecord class related to unknown native event types . 
 Commit message :  Fixes CAMEL - 2824 by avoiding deadlocks in aggregation logic . 
 Added dispose (  )  method for VideoRenderer in CallFragment . 
 Added parameter  - a /  -  - analyst and modified parameter  -  - bindAddress to enable OTP Analyst extensions
 Added spatial index for OpenCellID database
 Added codeOffset field to ImmutableInstruction20t class
 Added support for let and const declarations in externs . 
 Fixed issue with unresolved fx : id references in FXML files
 Fixed tile size in ModTextureAnimation to match changes in ModTextureStatic . 
 Fixed issue with garbled input causing incorrect response decoding . 
 Added contract annotation to method `deparenthesize` to ensure nullable return value
 Fixed startup mode handling for InfinispanSubsystemXMLReader
 Added license text for ViewConfiguration class
 Added method acquireWakeLockWithUid with additional parameters
 Added method to return CommonName from X509Certificate instead of IssuerName
 Fixed bug in acquireExclusiveLock (  )  method . 
 Fixed method signature in GrTypeDefinitionImpl to match the new body implementation . 
Added switch statement for PluginOperationResult . getAction (  )  in PluginCommand . printResults (  ) 
 Fixes issue where current page was not properly set when there are no more pages available . 
 Fixed copyright year for AnnotationJmxAttributeSource class
Added license notice to RetryingScheduledThreadPoolExecutor
 Added import for org . jf . smalidea . SmaliIcons and updated element icon for SmaliClass . 
 Changed handling of goToFile and reflowComment commands to fire corresponding events . 
 Added repetition of  ' A '  to test file for better performance . 
 Added method `recall (  ) ` to `Evaluation` class for calculating recall . 
 Added getNullables (  )  and getNotNulls (  )  methods to retrieve nullable / not null annotations for PsiModifierListOwner . 
 Added method  ' exists ( int userId )  '  to UserManagerService
 Added method remove ( Object )  to BroadcasterFactory
 Added method testModelParcelable to test the parcelability of the TestModel class
Parses wallet from binary serialization ,  handles corrupt wallets
 Commit message :  Fixes  # 12345 :  Added import statement for ApplicationManager . 
Added method signature changes and minor code improvements in NioEndpoint and SecureNioChannel classes . 
 Added support for building language console views with custom virtual files . 
 Fixed case sensitivity of column name lookups in ShadowSQLiteCursor
 Changed the implementation of the ExponentiallyDecayingReservoir class to use the forward - decaying priority reservoir sampling method ,  exponentially biased towards newer entries ,  as suggested in the provided paper . 
 Implemented changes to DbSqlSession to improve performance . 
 Added boolean isRoleSearchAsUser (  )  method and updated getter / setter for roleSearchAsUser field in JNDIRealm class
 Added default values for hidden ,  foreground ,  and background framerates . 
 Added float SCROLL_MULTIPLIER to ParallaxRecyclerAdapter for better scroll performance
 Fixed bug in MagicNumberCheck :  properly handling interfaces
 Added implementation of `matches (  ) ` method in CharMatcher
Fixed issue with unreasonably large weights / biases in DeepLearningModelInfo . java by adding new thresholds for mean_bias ,  rms_bias ,  mean_weight ,  and rms_weight . 
 Improved neighbor update percentage in PrepareContractionHierarchies from 40 to 10 for better performance . 
 Changed type of `nodeType` parameter in constructor from `IStubElementType` to `IStubElementType nodeType` to match the updated code . 
 Changed ParseTests . java to correctly handle unexpected end of file in message tag . 
 Fixed issue with event content not matching regular expression ,  now using whole content as startCommand instead . 
 Added method  ' getChangeLogParameters (  )  '  and updated corresponding code in Liquibase class
 Fixed issue with saving password in BaseAuthDialog . 
 Fixed bug in PyPropertyDefinitionInspection . java to correctly handle null call sites . 
 Added a new for - each loop to handle multiple resolve failures in UserResolverChain . 
 Changed how `setRefreshDrawable` method handles `RefreshDrawable` instance . 
 Changed the color for the EXACTLY_APPLIED case from blue to a darker blue - green color . 
 Fixed issue with permission rationale display in DexterInstanceTest . 
 Renamed collection from  ' wiki . users '  to  ' users '  . 
 Added method to handle network audit events outside of Hystrix context . 
 Added elevation profile for StreetVertex class
 Commit message :  Refactored VcsLogDataHolder and VcsLogRefresher classes to improve performance and reduce unnecessary data loading . 
 Renamed method `injectedToHost` to use shiftOffset instead of offset for better performance
 Fixed issue with incorrect option values in XmlReader . java
 Commit message :  Renamed ReaderTaskFragment . java to align with WordPress naming conventions . 
 Fixed issue with incorrect parameter order in getPeopleCountForLocalBlogId method
 Fixed bug in UUID generation for OfflinePlayer implementation
 Added blank header to list view with same height as mshot default height . 
 Rotated and scaled vertices for polygon . 
 Added test class for ResourceServerTokenServicesConfiguration
 Added volatile modifier to _readOnlyMap for better concurrency
 Committed revision 784710db93 with message :  Update ProxyServlet to ignore EofException . 
 Fixed bug in SearchController to correctly handle timerange type  (  # 102d597 )  . 
 Fixed empty relative path issue in TileAtlas . java
 Added new method `paintIcon ( Component ,  Graphics ,  int ,  int ) ` to IconUtil class
 Added support for annotated composite serialization of fields in subclasses
 Changed method signature to accept `Schema` instead of `JsonSchema` . 
 Fixed issue with finding documents in collection in CollectionAcceptanceTest . 
 Fixed layout issue in LocalFolderPickerActivity
 Changed interface to expose sender / receiver methods . 
 Parsed ExcludeClassInterceptors annotation and processed 97 method annotations . 
 Added new measure  ' connectionType '  and updated  ' getMeasures (  )  '  method to include it . 
 Added support for WebSocket version 8 in WebSocketServerHandshakerFactory
 Added new method for creating `ChangesBrowserNode` instances . 
 Added import for BlockingIOCometSupport and updated test to use it instead of Servlet30CometSupport in DefaultAsyncSupportResolverTest
 Renamed FakeTaskScheduler class from sample / src to rosie / src for consistency . 
 Added omitted methods to omittedMethodMap for pattern  ' pattern '  . 
 Fixed issue with tag filtering in ReaderPostListFragment . 
 Added support for custom frame APKs in AndrolibResources . 
 Added ability to handle raw syslog messages in SyslogEventHandler . 
Added import statements for StringTools class and updated code to read POS information from a plain text file in ManualTagger class . 
 Fixed formatting of timestamp in log message . 
 Fixed bug causing incorrect patch name display in editing stopped event . 
 Fixed issue with non - XML characters in JAXB marshaling
 Fixed issue with setting field values in ObjectMapper class . 
 Added activateOnCurrentThread (  )  to OSQLNonBlockingQuery for better performance
 Added new DiagnosticGroup  ' useOfGoogBase '  . 
 Added around advice to log method executions in org . jasig . cas packages
 Changed the behavior of the `killDbInSeparateThread (  ) ` method to wait for its completion before continuing with the test . 
Added support for empty  ' node '  attribute in EntityCapsManager . java
 Added new scope for improved variable handling in DefaultCodeFactory . 
 Added constructor for ComponentDao with System2 injection
 Added wheel and delta wheel values to MouseMotionEvent to improve debugging . 
Added method (  )  method to RequestSpec interface and updated LaunchConfigBuilder to accept a maximum content length
 Fixed bug in BSONSerde to properly handle missing fields in documents
 Added method to close model view in ModelNodeInternal for better performance
 Update ZoomManager with new scale and text wrap state after rotation . 
 Added loading of props files from classpath for DefaultAppCore . 
 Added method `attachRecords (  ) ` to Util class
 Fixed deprecated method in WebappClassLoader . java
 Added option  ' read - lookback '  to control read seed selection . 
 Moved listeners to a new array to improve performance . 
 Fixed issue with clearing app profiles for package updates
 Changed how QPtransform is implemented to use a new QPTreeTransformer object instead of the static QPTreeTransformer . valueOf method . 
 Fixed type encoding in ExpiresFilter initialization parameters . 
 Fixes bug in EqualsWithItselfInspectionTest by updating reference to correct method signature
 Fixed bug in PureFunctionIdentifier . Driver to properly handle null reportPath . 
 Added isLast check for proper binary message sending . 
 Fixed bug in OpenIdPostUrlHandlerMapping to correctly handle missing parameters
 Added singleton instance of Some < Void >  and method for returning it . 
 Fixed compile error in VcsInitialization . java by adding ApplicationManager . getApplication (  )  . executeOnPooledThread (  )  to execute (  )  method . 
 Added new benchmark method `startSillyBench (  ) ` and updated getCounter (  )  method to return int instead of long . 
 Added method  ' findAll '  to return all records of a single table in DataSupport class . 
 Added support for adding / removing regions in startup code
 Added import statements for Elasticsearch cluster routing APIs
 Fixed issue with getUpdateCenter (  )  method in PluginDownloaderTest
 Fixed incorrect tagging of  ' Tengo '  and  ' mi '  in Spanish . 
 Added method for setting text size in addition to text color
 Reverted changes to Robolectric . java to address compatibility issues with recent Gradle version . 
 Added historyRef and logger fields to HttpMessage class
 Added method to release session when view is detached
 Added empty `bounds` calculation for `calculateBounds (  ) ` method in `ActionToolbarImpl` . 
 Added ability to specify title for default value column
 Added builder class for SetHeaderHandler
 Added FAB_SIZE annotation with two new values  ( SIZE_NORMAL and SIZE_MINI )  . 
 Added test for iteratorRemove (  )  method
 Added new method to retrieve connected clients in ClientEngineImpl
 Added trace for JetPositionManager
 Fixed connection state reset method to use `zooKeeper . closeAndReset (  ) ` instead of `zooKeeper . getZooKeeper (  ) ` to avoid unnecessary connection establishment . 
 Fixed issue where transfer manager would throw an InterruptedException after reaching maximum retries ,  now it will throw a StorageException instead . 
 Fixed null pointer exception in Cache . java by updating the loader to return non - null values . 
 Changed method signature to throw a more specific exception for local storage . 
 Changed `setGridContentVisibility ( false ) ` to ` /  /  Only hide content if still in detail state .  if  ( mDetailRecord  !  =  null )  { setGridContentVisibility ( false )  ;  }` in `onAnimationEnd (  ) ` to improve performance . 
 Changed method signature to include rotation changes . 
 Added new GreeterGrpc . GreeterImplBase implementation to handle errors in sayHello method
 Fixed issue with retrieving deleted payment methods in BusinessFactoryBase . 
 Fixed issue with incorrect pointer sync in Clicker class
 Added new method createFileWriteAction ( File ,  String )  to IoActions class
 Added support for time counters in QueryFragmentCounters
 Fixed unused components in FirstPersonClientSystem for better rendering performance . 
 Fixed bug causing WebSocket session to remain open after close in TestWsWebSocketContainer . 
 CloneFields method added to deep clone object fields . 
 Changed behavior of `ensureCanCreateFile` to check if the file can be written instead of deleting it . 
 Added constructor for UsernamePasswordAuthenticationCredentials
 Added delay and error characteristics to mock implementations of API interfaces . 
 Added favoricon configuration
 Fixed month day to 15 for Calendar app compatibility . 
Fixed typos in BaseBinarySpec and BaseComponentSpec classes
 Added support for specifying target width and height in CameraLauncher . java
 Added listener for onCompletion event in OpenALMusic class
 Fixed return type of `RandomAccessFile` to include `ZipFile` for better inspection . 
 Revised SettingsImpl to only initialize editor settings for non - viewer editors . 
 Fixed null pointer exceptions in PrepareRequestTokenActivity by updating nullable fields to be null before accessing them . 
 Update SPACES_AROUND_KEY_VALUE_DELIMITER and KEEP_BLANK_LINES fields ,  and add new KEY_VALUE_DELIMITER_CODE field . 
 Added check for key equality in Scrubber . java to ensure consistency between data and index files . 
 Fixed naming exception in TransactionManagerTest
 Added toString (  )  method to ProgressManagerImpl for better debugging
 Added disableScrollingWhileRefreshing to PullToRefreshBase . 
 Changed how the `ModuleConfigurable` list is populated to use the `module` instead of `myModule` . 
 Fixed entity ID reuse issue in StorageManagerTest . 
 Fixed issue with separated dashes in RuleBasedCorefMentionFinder . java
 Added circular ripple selector and activated state to colored circles in ColorChooserAdapter
 ChangedPyStructureViewTest . java to include docstring for __module__ attribute . 
 Fixed null pointer exception in initialContext initialization
 Added support for Operation . CREATE in DocWriteResponseTests
 Added default value for call timeout to 60000 milliseconds in Operation . java . 
 Fixed issue with setting value to field for non - basic types in Form . java
 Added return statement in PageIndicator to handle null child views . 
 Fixed type resolution for missing types in TypeUtils . 
 Added exception handling for IOException in AbstractEmbeddedServletContainerFactoryTests
 Added new check for Closure rewrite class in ES6 transpilation passes . 
 Added concurrent map for storing current sessions . 
 Added license information and updated copyright year in package - info . java
 Added publicContentView field to BaseStatusBar for easier access to public version ' s RemoteViews . 
 Added ability to create user object from LDAP for new authenticated user . 
 Fixed issue with long - running test method . 
 Fixed bug in filename handling
 Added support for NS prefix in CamelNamespaceHandler for Spring integration
 Fixed issue with non - power - of - two framebuffer textures not being supported by the video hardware . 
 Added method `raw ( Class < E >  ,  String ,  Object .  .  .  ) ` to interface Queryable for entity type parameterization
 Fixed issue with standard extensions to ensure proper loading of javac from tools . jar . 
 Added support for timestamps in QueryableIndexStorageAdapter
 Added method signature for onCreateView (  )  with PrivateFactory injection  (  # 667302b )  . 
 Fixed bug in skipSomeNodes method to correctly handle inner polygons
 Fixed null pointer exception in MaterialDialog . build (  )  by updating null checks for content ,  title ,  items ,  customView ,  and adapter . 
 Added support for creating directories in DocumentsContract
 FixedDeprecatedMethod ( s )  in Intrinsics . java
 Fixed issue with ReplicaSet member change handling . 
 Added communication channel between scheduled job and sync manager
 Fixed bug in OpmlImportWorker constructor
 Added new constant  ' TRACE '  to Verb enum
 Added support for generic multi value types in ODocumentHelper . 
 Added annotation for ValueParameterDescriptor class
 Added import for org . jetbrains . annotations . NotNull
 Added new plugin registration logic
 Fixed exception message to include language name for better error messages . 
 Fixes issue with inconsistent path separators in UriUtils . java
Added new property  ' REUSE_RULES_CONFIGURATION_PROPERTY '  and deprecated the old one in CoreProperties ,  and updated Project to use the new property for determining whether to use external source for rules configuration . 
 Changed the layout margin of the FloatingActionButton to accommodate the navigation bar on devices with a navigation bar . 
 Fixed bug causing excessive memory usage by updating addAndGet method . 
 Added method to join list of strings into a single string for hidden drawer items  (  # 388 - 391 )  . 
 Commit message :  Fixes issue with cleaning up test server directory in InProcessServerControls . java . 
 Added support for HEAD method in RestController
 Changed the number of databases returned by `getImplementedDatabases (  ) ` from 7 to 8 . 
 Added method  ' error '  to MasqueradingPsiBuilderAdapter
 Fixed formatting issues in GetListActiveUsers . java
 Added method to get preferred focused component for DebugLogConfigureAction
 Added support for determining if device is encrypted with default password . 
 Added page size calculation for StoreCached class . 
 Added ability to mark change sets as re - ran with updated MD5SUM and ID columns . 
 Changed implementation of `ActionMenuPresenter . java` to support different action button behaviors . 
 Fixed issue with file type detection for MatcherImplUtil . java
 Resolved conflict for record  ' identity '  with current version 123 and other version 456 . 
 Added method to specify Downloader for Picasso . Builder
 Rewritten `performDexOpt` method in `BackgroundDexOptService` to use the new `true` argument for enabling dexopt . 
 Added check for script subtag in LocaleUtil . java to handle locale without a real language  (  # 565 ) 
 Changed how the `IndexReaderFactory` mock is created in `KernelStatementTest` . 
 Added support for arrays with one type argument . 
 Fixed license year in DnDManager . java
 Added Nullable return type for getDecompiledPsiFile (  )  method
 Added support for lambda expressions in predefined configuration . 
 Added new firewall rule for UID
 Fixed formatting issue in TpStatsPrinter . java
 Update to IssueComment :  added  @ CheckForNull annotation for userLogin field
 Added option to set the maximum duration for which a socket will reconnect after a failure  (  # 717ee8b ) 
 Added support for Java 14 module type
 ReplaceMessagesTest :  updated placeholder references in test cases
 Added `true` to `createRemoteProcess` to enable remote shell access
 Added logout message generation debugging statements for improved traceability . 
 Changed how arguments are mapped to resolved callables . 
 Changed how subtypes are logged from debug to info level . 
 Changed handling of synthetic classes to use `SyntheticElement` instead of `isSynthetic (  ) ` method . 
 Fixed type annotation for Reactor Fluxion in SimpleHandlerResultHandlerTests
 Fixed issue with visibility of result set panels . 
 Fixed layout bugs and improved performance by updating the scroll listener and margin bottom calculation . 
 Fixed bug in JSON serialization of enum values by using `name (  ) ` instead of `toString (  ) ` . 
 Fixed schema version number in DocSchemaInfo class . 
 Added log statement to track device list updates for own account . 
 Added new job prefix for Helios Solo Deployment Tests
 Fixed issue with null activity in updateFavourites (  )  method . 
 Added support for formatting docstrings in Python 3 . 10 with REST format  (  # PY - 8025 )  . 
 Optimized optimization method visitor with maximum instruction size limit increased to 5000 . 
 Fixed exception handling for FileLengthCheckTest
 Added new Sequencer implementation for direct access to the data structure without requiring a RingBuffer
 Added header removal logic in AbstractProxyServlet for improved proxy performance . 
 Changed how the `destFolderName` is set to handle special inbox name . 
 Added default value for secondsBetweenAutoSave to 5 . 
 Added exitWithSuccess method to handle help option
 Fixed null pointer exception in FList . java by updating equals method . 
 Fixed issue with truncate method in SinglePartitionSliceCommandTest
 Fixed issue where Watcher class was not properly handling missing file paths for sync accounts . 
Fixed issue with temporary folder configuration in GradleDistribution
 Added a new renderer type  ( 6th )  to the DemoPlayer class . 
 Added implementation of Iconable interface for LocalQuickFix . 
 Fixed bug in MessageHistoryServiceImpl by updating opSetSMS . addMessageListener ( this )  ;  to opSetSMS . addMessageListener ( messageSourceService )  ;  and removed unnecessary opSetIm . addMessageListener ( messageSourceService )  ; 
 Added new method to retrieve active loggers in factory
 Fixed issue with domain class processing . 
 Fixes bug by updating lengthOfChannel calculation to include entire message length ,  not just channel length . 
 Fixes IteratorIterable . java to use  ' if  ( iterator  =  =  null )  '  instead of  ' if  ( iterator  =  =  null )  {  .  .  . } ' 
 Added broadcast message for  ' Opping '  instead of  ' Oping '  . 
 Added support for manual density adjustment in ContactsContract . java
 Added method chaining for FindAndUpdateOperation
 Added governor arc labels to GrammaticalStructure objects in the basic graph . 
 Added WPWebViewClient to handle webview interactions
 Changed behavior of `hypot (  ) ` on ARM devices to match the expected result . 
 Update to SELinux enforcement status from 0 to 1
 Fixed issue where LatestVersionMatcher was downloading descriptor for requested version even when `latest . lowestStatus` was specified . 
 Fixed issue with empty array in StreamsTest . java
 Update FMLModContainer to use mod metadata version instead of internal version for mods without version metadata . 
 Fixes bug in RecyclerViewDragDropManager :  adds missing setter for wrapped adapter
 Added video link and updated documentation
 Added logging for compass sensor events
 Added converse method for JSR - 330 providers to directly implement Guice providers
 Added synchronized block around initialize (  )  method to ensure thread safety
 Fixed bug in IonBitmapCache to handle target width / height being 0
 Added property for initial token fetching . 
 Fixed bug in ImageDownloader . java by using AndroidUtils . executeAsyncTask instead of execute ( url )  to improve performance . 
 Added int [  ]   ' REQUEST_AVAILABLE_CAPABILITIES '  to CameraCharacteristics class . 
 Changed 1 line of code ,  added 1 import statement . 
 Fixed formatting and updated copyright year in ConsoleProcess . java
 FixedRatioWidget optimization :  updated inner widget to use raw widget instead of casting . 
 Changed layout direction and level of child drawables in LayerDrawable . 
 Fixed manifest file output path to prevent collisions
 Added support for CSIM app on UICC cards
 Added break statement in `cleanUpWriteBuffer (  ) ` to handle exceptions more efficiently
 Fixed buggy toString (  )  method implementation in DebuggerUtils to correctly handle all cases . 
 Added support for Grails tags in GSP pager  ( e . g .   <  %  g . link (  )   %  >  ) 
 Added new file to drop box for recovery log . 
 Added support for dynamic injection of broadcasters using named parameters
 Updating VERSION constant to 8 . 5 . 0 . 
 Fixed dependency inclusion for LoggingSubsystemTestCase
 Added preference for delay to start navigation in pedestrian mode
 Renamed SF to avoid confusion with incremental compilation
 Fixed splitter position updated to maintain visibility and attachments . 
 Fixed issue with node initialization and command classification in ZWaveConverterHandler . 
 Added enabled = false annotation to testTransferMigratedSubscriptionWithCTDInFuture for slow tests
 Added silence shutdown state to Gossiper . 
 Fixed bug in HystrixThreadPoolProperties . Setter (  )  method . 
 Added validation for sub - characteristics compliance and usability in migration v51 . 
 Fixes issue  # 123 :  Correctly handle database existence check in ONetworkProtocolDistributed . 
 Changed behavior of DoubleType to match FirebirdDatabase . 
 Added license notice to Whitelist . java
 Added save enabled false for ChatPager
 Update NotificationContentView . java to handle ForceSelectNextLayout
 Added suppression of unchecked warning for createEntity method
 Changed MediaMuxer . java to use temp . mp4 as the output file name instead of the default output . mp4 . 
 Commit message :  Fixes issue with Log . e (  )  statement in NativeBitmapFactory class . 
 Added copy of comparator for sorted sets in CollectionBinding . 
 Fixed method signature for getThroughput (  )  to match changed interface implementation . 
 Fixed validation message for private blogs . 
 Added method to request focus on the first editor in the composite . 
 Fixed bug in handling of multicast socket exceptions
 Added support for MessageDigestHashFunction with different algorithms  ( e . g .  SHA - 256 )  . 
 Commit message :  Added new method to save current text smoothing setting for later restoration . 
 Fixed issue with punctuation output in AnnotationOutputterITest . 
 Fixed encoding detection issue in Book . java
 Fixed bug where exchange client was not closed in some cases ,  now properly handling when all clients are complete . 
 Fixed bug in hasNextDispatch (  )  logic for ASYNC_END and UPGRADING states . 
 Added support for additional signing identities in iOS app signing . 
 Added new texture for PipeWorldRenderer . 
 Added Skylark method  ' nameInPlist '  to Platform
 Changed the annotation of a method to not be suppressed . 
 Update SamlIdPProperties to exclude unused import
 Added enablement check for history references in PopupMenuCopyUrls . 
 Added support for disconnected connections and updated state management in RemoteConnectionService . 
 Added storeName parameter to mutateMany method
 Added support for Java Time classes  ( LocalDate ,  LocalDateTime ,  ZonedDateTime )  as external classes for PrimitiveType enum . 
 Added custom fields to DefaultPaymentGatewayApi implementation . 
 Added getAssociatedObject (  )  method to BaseAdapterHelper
 Added new fields to the Calendar provider :  ACCOUNT_NAME and ACCOUNT_TYPE . 
 Added maturity level to GuardedByChecker
 Changed method signature to include additional package ,  uid ,  and common process name for logging purposes . 
Refactored ClassReflection class to use type - safe generic method for creating instances  (  # 1234 )  . 
 Added ability to select up to 100 folders in VcsStructureChooser dialog . 
 Added  ' try '  keyword to CheckUtil class
 Changed how `match . getMessage (  ) ` is stored in the database ,  now using `StringUtils . abbreviate (  ) ` to truncate long messages . 
 Changed the size of the payload buffer count in NewBufferMessageFlyweight from 4 to 6 . 
 Fixes matrix multiplication for negative values . 
 Fixed compatibility issue with Java 9 by adding colon to URL_PKG_PREFIXES property . 
 Fixed null pointer exception in FetchBlogListWPOrg . java by updating the validation of the XML - RPC response . 
 Fixed line breaks and slashes in JavadocOptionFileWriterContext to conform to JavaDoc options file format . 
 Added deprecated constructor for ExponentialBackoff with baseSleepTimeMs and maxAttempts parameters
 Added GroupNames . DATA_FLOW_ISSUES to the inspection ' s group display name
 Added template name to GoCreateFileAction class
 Changed behavior of EditorTextField to release editor on project closing . 
 Added constructor for DeleteIndexedScriptRequest
 Changed how JDK classpath is determined for Java 9 +  . 
 Added method to consume LookupElements from contributors in a more efficient manner . 
 Commit message :  Refactored method `XmlLog . formatXML (  ) ` to improve readability and reduce code duplication . 
 Fixed bug in SoundTest . java regarding incorrect license text . 
 Commit message :  Added import for InjectedLanguageManager to handle injected fragments in PsiFiles . 
 Added import for ClusterState and updated list method to include node IDs in the request
 Fixed issue with id (  )  method
 Fixed issue with ticket expiration in GenerateServiceTicketActionTests
 Added constant for Android ID  ( a 64 - bit hex string )  for device identification . 
 Added new TransitionSet class method to set start delay for first transition in sequential playback . 
 Added new regular expressions for identifying multi - word phrases in Spanish . 
 Added getStatusMessage (  )  and getStatusMode (  )  methods to EditorStatus class
 Fixed comparison method to use Float . compare (  )  for more accurate distance comparisons . 
 Added region download names for live updates in ReportsFragment . 
 Fixed issue with empty templates list display in ListTemplatesHandler . 
 Added import for Desktop class in DomUtils . java
 Fixed type for polling consumer in ScheduledPollEndpoint
 Fixed handling of unsupported content types in HttpConnection . 
 Fixed dialog icon in ShutdownThread . 
 Fixed layout issue in GoCommandConfigureDialog
 Fixed issue with auto - zoom not working correctly when switching between map types . 
 Fixed issue with Pattern . quote (  )  method to correctly handle string literals . 
 Added executeActions (  )  call to handle enqueued actions on detached views . 
 Fixed filter suffix check to handle missing  '  )  '  in Emmet templates
 Added log warning for end - user WS - Endpoint exceptions . 
 Added activation energy requirement for TileMachine
 Fixed issue with double slashes in FtpUtils . java
 Changed FontAwesome class to include new icons :  CHECK and EXCLAMATION_TRIANGLE . 
 Added support for bulk write operations in MongoCollection
 Added empty line array initialization for FakeFile class
 Fixed token return type in Check . java to match TokenTypes . 
 Added extra  ' targetFile '  to intent for back button navigation to display activity . 
 Initialized DataSourceInitializer with updated properties and data source . 
 Changed how `hex [ c ( 8 , 1 , 1 , 7 )  , 1 ] ` is handled in `RapidsTest . java` . 
 Added new version check for Atmosphere ' s JavaScript Client in AtmosphereFramework . 
 Added  @ Deprecated annotation to CarbonClient constructor
 Added check to ensure  ' type '  is always  ' LITERAL '  or  ' INTERPOLATION '  in TokenValue class
 Added support for SVG images in HTML formatter
 Added unit type to candidate classes for Kotlin Runtime Library Core Util . 
 Added import for ComException
 Added handler for search text changes to improve performance
 Renamed MocoJsonSocketHttpRunner to MocoJsonSocketRunnerTest for test purposes
 Fixed exception handling in OServerCommandGetDatabase to correctly handle non - existent clusters . 
 Added idempotent `run` method to ExecutionList . 
 Added new notification dispatchers and removed old ones for improved notification handling . 
 Added import statements for Android graphics and drawable classes
 Fixed method signature to match updated class structure
 Fixed buggy line number formatting in DebuggerTestCase . java
 Fixed bug in decryption logic for AESCrypto class . 
 Update PRuneCutoffDays from 45 to 15 for SecretContentDAO
 Added caching for resource retrieval in RequestServer
 Added method `CometProcessor . process (  ) ` to extend servlet functionality
 Changed CompilerOptions to include new `setUseTypesForOptimization` and `setRewriteFunctionExpressions` methods . 
 Fixed issue with dynamic records in Xa log . 
 Fixed formatting issues and updated copyright year from 2009 to 2013 . 
 Added  @ Deprecated annotation to method signature in ArchiveHandler . java
 Added method  ' classLocation '  and  ' joddLocation '  to RuntimeUtil class
 Added Window . FEATURE_ACTION_BAR feature to ActivityController
 Fixed issue with ActionMode not being initialized correctly
 Added support for guessing  ' apply '  render state values based on defaults in Material . java
 Added removeLongPressCallback (  )  method in View class
 Added check for suspended applications and disabled account management for specific types of accounts . 
 Added setup method to AbstractTestTag class
 Added deployment with a single resource
 Reordered result set viewer to improve performance . 
 Added support for generating stubs for floating - point and character literals
 Changed method signature for `getSeaLevel (  ) ` to include block measurement . 
 Fixed empty street name and reference handling in MenuTitleController . 
 Fixes view fetcher to properly handle deprecated method calls
 Fixed issue where span had no matching tag from requested groups ,  now logging error instead of info . 
Fixed issue with excessive output in FullyExecutableJarTests . 
 Added support for Thermostat Mode Set command in ZWaveThermostatModeCommandClass . 
 Added public static final CREATOR field to ParcelableGenerator class
 Added testDateLiteral method to TestExpressionInterpreter
 Added explicit delegate equality check for IOSMethodBinding # overrides (  ) 
 Added listener for write operation in NettyAsyncHttpProvider to handle progress updates
 Added parameter  - Dprofile . groovy . compiler = true for profiling Groovy compiler
 Added default value to boolean properties in Defn interface
 Added check for focused stack before throwing IllegalArgumentException for invalid task . 
 Fixed method name collision issue in StructStrategy
 Fixed serialization of PlistDictionary with integer value 42 . 
 Changed import statement for Jimfs to include the Configuration option for Unix . 
 Added support for new resolutions and scaler available processed sizes in CameraProperties . 
 Fixed indentation in RESTDocsGenerator . java to match AsciiDocGenerator . 
 Added new privilege  ' OWNERSHIP '  to HivePrivilege enum
 Fixed formatting of TimestampWritable objects in SerDeUtils . 
 Added support for canceling editing with ESCAPE key in DarculaLaf . 
 Fixed issue with ProcessCanceledException handling in JobUtil . 
 Added import for javax . script . ScriptContext and modified method to clear bindings in ScriptContext . 
 Added logging for malformed headers
 Fixed bug in evaluate method to provide more informative error message
 Fixed null pointer exception in ContentStreamingRequestAction by adding a null check and using the touch (  )  method to update the received list . 
 Added two pixels to the line number area width in EditorGutterComponentImpl . 
 Added logging config loading and uncaught exception handling for Android . 
 Added constructor for ORecordBytes to accept an ODatabaseRecord argument . 
 Fixes issue with singleton annotations on constructors in GraphAnalysisInjectBinding class . 
 Added default value for max_hint_window_in_ms to 1 hour . 
Fixed issue with nearbyPoints size in VpTreeNodeTest . java
 Added default module  ' CORE_GAMEPLAY_MODULE '  to configuration subsystem . 
 Implemented new length parameter for session ID generation
 Added convenience constructor for BasicBSONObject
 Commit message :  Added support for TerrainGrid initialization with scene location . 
 Added optimization to set content length on writeFinal if no data has been written . 
Fixed issue with navigation menu items in SherpafyCustomization . java . 
 Commit message :  Fixes issue  # 1234 by updating the RecoveryMode to be properly set from configuration . 
 Fixed tab width calculation in Utils . java
 Added new test method for inserting a phone entity and testing the relationship with the owner person entity . 
 Added check for user initialization before remounting external storage for changed permissions in PackageManagerService . 
 Added exception handling for corrupted project files
 Changed method signature to return `EBoolean` instead of `void` . 
 Fixed issue with read - only files in DiffContentUtil . java
 Commit message :  Fixes issue with YOLO suggestions not being displayed in EyeemActivity . 
 Added ignore check for satellite sources
 Added new animation for message bar in MessageBarUtils
 Added support for handling EOFExceptions in the SnakeTimer . 
Fixed issue  # 1234 by adding a new method to InvalidJsonException to return the problematic JSON if available  (  # 5c660b3 )  . 
 Fixed buggy assignment operator in GatedStreamSinkChannel . 
 Changed behavior of `SpdySessionHandler` to call `trySuccess (  ) ` on `closeSessionFuture` instead of `setSuccess (  ) ` to improve performance . 
 Fixed issue with interpreting HTML characters in topic change event report message . 
 Added `propertyChanged (  ) ` method to DerivedStringProperty
 Changed behavior of search function to include rooms starting with specified name . 
 Added support for gzip compression in StaticS3FirehoseFactory
 Added support for pluralized service types in ServiceDiscoveryResult
 Fixed column size increased from 100000 to 10000 for MultiValueDictionaryTest . 
 Added Executors . newSingleThreadExecutor (  )  to ClientAuthInterceptor for improved performance
 Fixed dependency management for  ' httpuv '   ( 1 . 3 . 3 )  ,   ' caTools '   ( 1 . 13 )  ,   ' mime '   ( 0 . 3 )  ,   ' jsonlite '   ( 0 . 9 . 16 )  ,  and  ' xtable '   ( 1 . 7 )  . 
 Fixed issue with null pointer exception in WifiStateMachine . java when handling CMD_GET_CONFIGURED_NETWORKS command . 
EmailAlarmCallback updated to include check_result . resultDescription in the email subject . 
 Fixed issue with countdown in ClusteredEventBusTest
 Fixed test case execution order to improve test stability
 Added ExasolTableBase as the child type for ExasolSchema
 Added support for extending query executors through the use of extensions . 
 Added method to set name override for fragments . 
 Changed file from 238 bytes to 238 , 20 bytes . 
 Added support for mapping fields with  @ Column annotations in Java class . 
 Added unique constraint change generator for backing index  ' backingIndex '  with name  ' backingIndex . getName (  )  '  . 
 Fixed bug in writeByte (  )  method to correctly handle null values . 
 Fixed issue with incorrect password in Hazelcast client initialization . 
 Changed behavior of `queryExtensions` in `PipelineRepositoryTest` to use mock instead of real implementation . 
 Fixed issue with icon path in PreviewSettingsTopComponent  (  # eeb1c527f )  . 
 Fixes ParseException to better reflect its purpose and avoid confusion with EvaluationException . 
 Changed 14 lines ,  added 14 lines in b / sql / src / test / java / org / cratedb / service / InformationSchemaServiceTest . java . 
 Changed behavior of TaskViewServiceTest to use TaskConfigProperties instead of hardcoded property names . 
 Added CONNECTIVITY_MANAGER_SINSTANCE excluded ref for leak due to sInstance field . 
 Added method to get ServerAddress in SingleChannelMongoClient
 Added handling for tab keypress in ContactListSearchKeyDispatcher . 
 Added new methods for replacing strings with icons
 Improved cache for built - in types in Python 3 . x
 Fixed bug where clicking on a scrolled - off view would cause an invalid item click . 
 Fixed bug in Node accessor to return correct node instance . 
 Added session id generator method in ManagerBase
 Fixed method signature for LocalFileSystem . open (  )  to match the updated implementation . 
 Added new method  ' systemReady (  )  '  and updated userListFile initialization in UserManagerService . 
 Fixed HTML escaping issues in SwingSpyPanel . 
 Added check for package name in notification filtering
 Fixed bug in handling of permissions results . 
 Fixed issue with arrival times for first stop in Timetable . java
 Fixed bug in NavDrawerData constructor to include downloaded items count . 
 Fixed issue with store definition not matching existing store in cluster . 
 Added new caching property KW_CACHING and updated existing property KW_COMPACTION to be consistent with the new naming convention . 
 Added getter method for isDefaultReadOnly (  )  to enable migration to Java 14
 Added support for JPEG image capture
 Added timeout util method for adjusting sleep time
 Added new method to set interest ops for NioWorker
 Fixed issue with NativeOsmandLibrary preloadLibrary (  )  method ,  improved library loading process . 
 Added support for query caching in SearchQueryQueryToolChest . 
 Added support for debugging in JSPX files . 
 Added onChange (  )  method to notify listeners when a Realm instance has been updated . 
 Fixed directory creation issue in CsvReporterFactory to ensure correct metric writing . 
 Added soft block detection for BlockFluid ,  BlockTallGrass ,  and BlockFlower . 
 Fixed copyright notice in DependencyInjectorGenerator class
 Closes the underlying file handle
 Added new implementation for JpsIdePluginManagerImpl . 
 Fixed issue with alarm callback service in AlertsController . 
 Added ignore list for plural forms of  ' Kilogramm '  and  ' Uhr '  in German . 
 Added method  ' setSlowSqlMillis '  and  ' setLogSlowSql '  to StatFilter ,  and updated isLogSlowSql (  )  to return a boolean value . 
 Commit message :  Fixes bounce behavior for offline messages of type normal ,  groupchat ,  or headline . 
 Added method to handle zero - length frames . 
 Added JSPX custom search helper
 Added server . openDatabase (  )  method to create database instance for binary protocol requests . 
 Added SSL client authentication to Server class
 Added import for org . jetbrains . annotations . NotNull in AbstractJavaToKotlinCopyPasteConversionTest . 
 Added removeWallet (  )  method for removing a wallet from the chain . 
 Added filter to control panel entry factory to only include entries with the correct object class . 
 Fixed SSL validation issue by handling recoverable exceptions in SslValidatorDialog . 
 Added implementation for interface flag in V3ClassAdapter
 Fixed null pointer exception in PropertiesExpander constructor
 Added new `Person` constructor to `PersonSessionBean` to create a new person instance with the same name and age as the original person object . 
 Added case KeyEvent . KEYCODE_D to handle delete action
 Added new endpoint for LinkedIn API authentication
 Changed line 84 from ` -     return null ; ` to ` +     return new FileSystemMasterService . Processor < FileSystemMasterServiceHandler >  ( new FileSystemMasterServiceHandler ( this )  )  ; `
 Added loadMetadataSuceeded variable to handle directory creation failure in FileSystemMaster . 
 Fixed NoSuchInputTypeException error message in Inputs . java
 Added mapColorForTab (  )  methods for tabs 2 ,  3 ,  and 4 . 
 Fixes NPE in connectAndSendHeaders (  )  for ServletClientSocketChannel . 
 Added Apache License 2 . 0 to IFileDownloadServiceHandler . java
 Fixed issue with commit / rollback method in InMemoryTitanGraph class . 
 Added synchronized checkResources method to avoid unnecessary redeployment and reloading of web applications . 
 Fixed issue with home button behavior in TaskListActivity
 Changed the value of `SOCKET_TIMEOUT` from 1000 to 5000 . 
 Changed type signature for ModelType to include the length of the model bytes . 
 Sorted filters by order in RestController . 
 Added import for Android Support Annotation  @ NonNull and  @ Nullable for BottomSheetBehavior . 
 Fixed issue with saving sharing settings on pause
 Fixed bug in fillMessageMutation method
 Added binding listener to ClusterClient for multi - paxos server updates
 Changed `JOBJAR_PATH` to `JAR_PATH` in `TestSharded . java` . 
Added method to register GoConfigWatchList instance in constructor  (  # 640e317 )  . 
 Added constructor for TestType with parameterized annotations
 Fixed issue with empty paths and absolute paths in CommonUtils . java
 Fixed bug in lastSelected retrieval to correctly handle empty selection . 
 Changed the `finally` block to properly unlock and stop training the model after it is created . 
 Fixed issue with tokenization in ConfusionRuleEvaluator class  (  # 99 )  . 
 Fixed issue with maximum connections per node in AdminConnectionVerifier
 Added support for handling quotes in SQL auto - indentation . 
 Added ability to export chat to file with specified name . 
 Fixed method signature for getEnergyStored (  )  to match battery API . 
 Added NativeManagementAddHandler and HttpManagementAddHandler for on - demand services in DomainModelControllerService . 
 Fixed type keyspace for drop function statement
 Added getAudioState (  )  method to return null when mCallAudioState is null
 Added constructor for LiquidTank with capacity parameter
 Added 0 . 3f relative size span to message bodies for improved readability . 
 Changed the behavior of handling JVM arguments to include all non - empty arguments ,  rather than only those starting with  - Xbootclasspath :  or  - javaagent :  . 
 Added hashCode (  )  method for KSMetaData
 Fixed bug in sendfileData calculation
 Added new appearance options for main tool bar ,  status bar ,  and navigation bar ,  and updated existing appearances for window animations and memory indicator visibility . 
 Fixes issue with  # ^ symbol in RT . print method
 Changed usage of  ' Use super method of '  and  ' Change usages of '  in GrIntroduceParameterHandler to  ' Change base method '  and  ' Change usages '  respectively . 
 Added new LineString constructor to AbstractConverterTest for improved performance
 Added new MultipartConfigElement to ServingLayer for better performance
 Added new rendering support for TextEditorProvider
 Fixed issue with unstable beta releases in ShowsActivity . 
 Fixed bug in getChildCount (  )  method
 Changed behavior of space key to immediately switch to following state instead of waiting for 0 . 5f seconds . 
 Added spacing calculation for seek bar ticks
 Added support for Spring Roo entity annotations
 Changed implementation of `ActionUrl` class to support creating groups for actions . 
 Added doLexingOptimizationCorrectionCheck to true to ensure consistent re - lexing of lazy parseables . 
 Added nullability checks for subcolumn and custom index parameters in CFMetaData constructor . 
 Fixed issue with R22 of the SDK by removing unsupported option  -  /  /   @ mosabua . 
 Fixed formatting issues in Invocation . java
 Fixed issue with waited request handling in OHazelcastDistributedDatabase . 
 Added flag for wiping data and factory reset protection in DevicePolicyManager . 
 Added new method to generate JSON properties
 Added CDI support to IntegrationTest class
 Added ChunkSatelliteView to SourceSatellite class
 Added deprecated method  ' addArtifacts '  in favor of  ' addTransitiveArtifacts '  to prevent increased memory use . 
 Reinitialized ConversationActivity and ConversationFragment with updated messages . 
 Added support for Android ' s theme picker in SeriesGuidePreferences . 
 Changed how much data can be read at once from the data source . 
 Fixed type safety in OnErrorThrowable . java by casting null values to correct type
Added default value to property  ' camel . usdDot '  and updated related code to enable DOT file generation . 
 Added support for notifications on device provisioned devices
 Fixed bug in Parallel Gateway activity ,  now properly handling AND - split and AND - join behavior . 
 Fixed SecurityUtil . remove (  )  call to match latest API . 
 Added outer interface to VPN configuration in Vpn . java
 Added import for Objects utility class
 Added author Remy Maucherat to CometEvent interface
 AbstractMessageTask :  updated method signature for getDistributedObjectType (  ) 
 Fixed issue with null pointer exception in AbstractCamelController . java by updating the  ' title '  field to be nullable . 
 Added exception handling for UnresolvedAddressException in AbstractEpollChannel . 
 Fixed bug in handling of non - XmlTagChild elements in XmlAspectImpl . 
 Added support for unlimited strength ciphers
 Fixed buggy inheritance resolution for classes with circular references
 Fixed bug in CommonXml . java to properly handle restricted path names . 
 Added import for MoreObjects in Bindings class
 Added new native ARGB conversion method for OpenGL
 Changed how the `ButtonGroup` is created to ignore custom creation if the field is bound and not a `ButtonGroup` . 
 Added support for logging all commits ,  including those in index and stash ,  using the ` -  - all` parameter . 
 Fixed exception message for tenant - specific catalog loading
 Added projectKey to ScanIssues class for improved reporting
 Added property to enable clustering  ( CLUSTER_PROPERTY_NAME  -  >  CLUSTER_PROPERTY_NAME )  . 
 Added StringLiteralQuotesAnnotator to the list of annotators . 
 Stopped disconnecting debugger and closed it . 
 Added delay before clearing state for auto pay off due to lack of correct synchronization . 
 Added while loop to improve performance of setIfAbsent method
 Added sleep interval to improve test performance  (  # 180 ) 
 Changed DataForm class to include a new `type` field . 
 Added new preference test cases for _prefer_node : somenode and _prefer_node : server2
 Added check for empty activities list in ActivityStack . 
 Fixed issue with file tree implementation in PathKeyFileStore
 Added new panel height calculation for sliding up panel layout
 Changed code to add a new line of code to handle partial mask matches in the `QTestUtil` class . 
 Added check for package existence in getApplicationEnabledSetting (  )  method
 Added / updated static variables for ThreadExecutor class . 
 Fixed compatibility issue with IBeaconDataFactory . 
 Changed the default priority value from a string to an integer in the `NotifyMyAndroidActionService` class . 
 Added license header and updated copyright year in CursorException class . 
 Added new configuration options for SURROUND_SELECTION_ON_QUOTE_TYPED and AUTOINSERT_PAIR_BRACKET / QUOTE . 
 Fixed issue with abandoned connection cleanup thread in CleanupListener class
 Fixed issue with RedisException handling in RBatch . execute (  ) 
 Changed test to pass :  added new chat test . 
 Added support for window state animations on the default display . 
 Sorted issues by assignee . 
 Added support for multiple values in request headers for  ' Accept '  and  ' Content - Type '  headers
 Added KeyguardViewStateManager to KeyguardHostView for improved performance and reliability . 
 Fixed bug in TreeExpandableItemsHandler related to incorrect handling of expandable items . 
 Added method  ' getContext (  )  '  to FakeDialog class
 Deprecated method  ' getDefault (  )  '  in  ' BroadcasterFactory '  replaced with  ' getBroadcasterFactory (  )  '  in version b26e80d . 
 Added uppercase naming convention for constant names in PyIntroduceConstantHandler . 
 Added getStreamIdentifier (  )  method to SctpPayload class
 Updated comment status in SQLite for successfully moderated comments . 
 Fixed logging method signature to match updated interface contract . 
 Fixed issue with deployment handling in DomainXml class . 
Added polymorphic cluster IDs sorting and copy for OClassImpl  ( ebaf437 .  . d370c3e )  . 
 Changed MessageTest to include a new test method to test the conversion of a date object to a DateTime object . 
 Added CameraRuntimeException for unknown camera device errors
Added pattern matching for images in Utility . java
 Added weblogic workaround for Environment . detectClasspath
 Fixed event naming for  ' login_magic_link_ '  events . 
 Fixed error message for missing AtmosphereHandler Service annotation
 Fixed exception handling for FailedToCreateRouteException in HttpClientRouteTest
 Fixed bug causing incorrect handling of excluded URLs in Maven Root Model Adapter . 
 Fixed issue with empty permissions list in Facebook . java
 Added content text to notifications for better readability . 
 Added method to check for existing world before loading it
 Fixed deployment resource class to include updated method signature
 Fixed issue with empty intermediate points list
 Initialized BouncyCastle security provider for cryptography support . 
 Added  ' lshistory '  command to ClearCaseRepository class
 Changed `empty_view` layout ID to match new resource identifier . 
 Fixed rendering issue in LwjglCanvas by calling resetGLObjects (  )  after createContext (  )  . 
 Changed how Kuromoji is integrated in tests to use a temp dir instead of a classpath plugin . 
 Changed format options for Cucumber reports . 
 Fixed bug in accumDots method for dense matrices
 Changed the repository for artifact uploads . 
 Fixed buggy resume / cancel logic in AtmosphereResourceImpl . 
 Fixed issue with incorrect cookie generation in SendTicketGrantingTicketActionTests . 
 Added new BasicHeader with random value for  ' relevantHeaderName '  in ContextAndHeaderTransportIT . 
 Added templateDir field to DjangoProjectCreationTest class
 Added cancel (  )  method with exception handling for DaemonProgressIndicator . 
 Renamed thread naming logic to use `setName (  ) ` instead of `setThreadName (  ) ` for improved security . 
 Added regex pattern matching for Uri in FakeHttpLayer
 Added support for NetRCCredentialsProvider in GitCloneFunction
 Added property minPoolSize to HikariConfig
Fixed JoyInput to properly handle GLFW_JOYSTICK_LAST
 Added support for composite elements in ASTWrapperPsiElement . 
 Fixed buggy assertion in XmlElementFactoryImpl . java
 Fixed density - dependent value inflation for DrawableContainer . 
 Fixed hash code calculation for WebAuthenticationDetails class . 
 Added method `userInput ( String message ,  List < String >  validResponses ) ` to GrailsConsole class . 
 Added shortenPathWithEllipsis method to CommandLineUtil for improved readability . 
 Added new  ' NONE '  rules category
 Added serialVersionUID to ToroTaskExecutionException
 Commit message :  Changed edge constraints and added vertex properties in SQL Create Edge Test . 
Added new rules to EngineRuleSet and HostRuleSet classes in java / org / apache / catalina / startup directory . 
 Added method `getPercentageContended (  ) ` to calculate percentage of contended operations . 
Fixed listener interface implementation in EmojiconGridFragment and EmojiconsFragment . 
 Added method  ' applyLearningRateScoreDecay (  )  '  to RNTN layer . 
 Changed OracleDatabase . java to log a different message when compatibility mode is not enabled . 
 Added method to determine if chart can be scrolled when embedded in ViewPager . 
 Added line break detection in BookmarksActivity
 Added new action to Banner component
 Fixed handling of non - streaming resources to include output junk . 
 Added support for loading response body as a byte array
 Added new constant ACTION_SHOW_REGULATORY_INFO
 Added import statement for Java File API
 Added listener for on - demand authentication state changes . 
 Fixes type casting issue in AShader . java by changing  ' ShaderVar v  =  new ShaderVar .  .  .  '  to  ' ShaderVar v  =  new ShaderVar ( DataType . VEC3 )  .  .  .  ' 
 Added ability to specify alternate access level for getter methods
Fixed NotImplementedException in JsContext . java to UnsupportedOperationException for improved compatibility . 
 Changed a single line in `SimpleDaoTest . java` to add a new test method for chain insertion . 
 Added new `clear (  ) ` method that removes all entries in the set without concurrent access issues . 
Added isDone (  )  method to TaskState enum
 Added layout method to adjust child views '  positions based on screen size
 Fixed issue with edge - based graph conversion in OpenStreetMapGraphBuilderImpl
 Added CharsetEncoder factory for improved performance
 Changed the `TripTimeSubsetCache` field from `transient` to `transient TripTimeSubsetCache` to make it a separate field . 
 Changed behavior of `repeat (  ) ` method to handle null input . 
 Added DKV . put (  )  call to update model state after removing and adding vectors to the training set . 
 Added import statements for SampleRoboApplication_ and updated imports to align with latest API changes . 
 Added setCompoundDrawable (  )  method to set target density for drawables in RemoteImageLoaderTextViewAdapter
 Added new gauge for BufferPoolMetricSet :   ' used ' 
 Fixed bug in TagFileProcessor . java by ignoring MalformedURLException when tagFileJarUrl is null . 
 Fixed bug in Aeron . ClientContext constructor . 
 Fixed null pointer exception in PrestoS3FileSystem . java by updating import static com . google . common . base . Throwables . propagate ;  to import static io . airlift . units . DataSize . Unit . MEGABYTE ; 
 Added title bar for PreferenceScreen
 Added support for executing abituary HQL queries
 Fixed filter to only run tests
 Changed method comparison to use `getX (  ) ` ,  `getY (  ) ` ,  and `getZ (  ) ` to compare coordinates instead of directly comparing integers . 
 Commit message :  Added new implementation for Folding Cell in Android platform . 
 Added mediaFile field to WPEditImageSpan for improved performance
 Added commit (  )  method to Neo4j2Graph class
 Fixed issue with query string parameter handling in AtmosphereFramework . 
 Added new method `forcePublish` to Sequencer ,  which sets the claim strategy ' s sequence and publishes the next event . 
 Fixed null pointer exception in AnnotatedElementUtils . searchWithGetSemantics (  )  by adding a check for empty annotations . 
 Added replacement of Windows path delimiters to prevent conflict with Lucene escape character
 Commit message :  Fix for self - hosted blog edge cases in StatsUtils . java . 
 Added support for loading multiple classes with the  @ SpringBootTest annotation
 Added default LZ4Compressor implementation for faster compression . 
 Added import for PyUserSkeletonsUtil and updated logic to skip directory containing user skeletons in PyTreeStructureProvider . 
 Commit message :  Fixes issue where static access to HystrixRequestLog fails without context . 
 Added newSet (  )  method to SearchAction to return a set of possible fields based on the input parameters . 
 Reverts change that causes critical processes to be killed during setup . 
Update ODatabaseRecordWrapperAbstract and ODatabaseWrapperAbstract to use ZIP compression with configurable level  (  # 583b215 )  . 
 Added pointerIndex to CustomViewAbove to handle drag events more efficiently . 
 Added support for in - memory channels in ExecutionVertex . 
 Added test suite name attribute to JUnit report
 Added experimental data reduction proxy setting to WebView . 
 Added support for probing lights in MultiPassLightingLogic
 Added test to verify supportsExpectingStatusCodeWhenAuthenticationError (  )  with an error message
 Fixed issue with loading forgeChunkLoading . cfg file . 
 Fixed null pointer exception in PredicatesTest . testInNullWithNullArgument (  )  and PredicatesTest . testInNullWithNullArray (  )  methods . 
 Added unsorted list generation method to SortActivity
Added serialVersionUID to CommandFailureException
 Added context variable for WebViewRenderer
 Fixed license year in DualView . java
 Fixed null pointer exception in `after (  ) ` method
Added import statements for mockito and improved code readability  (  # 88f2590 ) 
 Added method signature for ListenableFuture . addListener ( Runnable ,  Executor ) 
 Fixed issue with package version code comparison in PackageManagerService . 
 Added import for com . codahale . metrics . health . SharedHealthCheckRegistries
 Added support for mining well orientation in BlockMiningWell . 
Fixed issue with RippleDrawable in TaskViewHeader . java . 
 Added support for new HTTP parser configuration options . 
 Added  ' accessibility_display_inversion_enabled '  and  ' accessibility_display_daltonizer_quick_setting_enabled '  settings . 
 Changed how password is handled in JcrEndpoint . 
 Fixed bug causing incorrect chunk length calculation in AjpServerRequestConduit . 
 Added logic to check the correctness of commitBlock when retry is called . 
 Fixed formatting in App . java for improved readability and compatibility . 
 Changed how the `tests . bwc` property is handled in the `ReproduceInfoPrinter` class . 
 Fixed issue with panel hiding when slide animation is enabled
 Fixed issue with null tile at end of list . 
 Fixed bug in ECKeyTest . java by modifying encrypted private key to break encryption reversibility . 
 Allocate direct ByteBuffer for reflection allocation . 
 Changed interface names in JavaScript bridge to match new class names in Java . 
 Update MasterWorkerInfo to setLastUpdatedTimeMs instead of updating last updated time directly
 Fixed issue with TTS requests and improved performance by reducing unnecessary alerts . 
 Commit message :  Fixes  # 994be4c72 :  Added getPercentLayoutInfo (  )  method to return updated layout info for PercentFrameLayout . 
 Added method to get variable number of parameter values from audio hardware
 Added currentThread . setName (  )  method to log version and build number for each thread
 Fixed issue with getResource (  )  method in AbstractTestResourceSet . java
 Added support for injecting Boolean values in InjectableObjectFactory . 
 Fixed NullPointerException in for loop in CodeSmellDetectorImpl . java
 Added comparison logic for primary blog in SitePickerAdapter
 UIPrefsAccessor :  updated surround_selection preference from  ' Quotes  +  Brackets '  to  ' Quotes  &  Brackets ' 
 Added check for default user agent string in ContentView . 
 Fixed bug where KeepInSync was not being saved correctly
 Added field mapping for MappedContainerTypeHandler
 Added component - level dependencies for JvmLibrarySpec
 Added replacement for EDH in handshake algorithm
 Added constructor for Decal class
 Fixed NPE when going back to home activity
 Added method dispatchMenuItemSelected to MenuItemImpl
 Added NotificationsListPanelWrapper for improved notification rendering . 
 Added dependency on JUnit Test Suite Binary Spec for test tasks
 Rebuilt polling table with new items . 
 Fixed issue  # 12345 :  Added missing  ' msgid '  field in response . 
 Fixed buggy code in AbstractRecord . java ,  now correctly setting all changed values when a field is modified . 
 Added Linux and Mac paths for BrowsersConfiguration class
 Fixed issue with Recents not being dismissed when PIP moves to fullscreen . 
 Added property  ' org . opensolaris . opengrok . history . NumCacheRenamedThreads '  to configure the number of cache renamed threads . 
 Added registration for new ores :  stone ,  cobblestone
 Fixed formatting issue in TspAnt . java
 Fixed navigation drawer layout issue by updating target view parent check . 
 Added handlePostDhcpSetup (  )  method to restore power mode after DHCP setup . 
 Fixed issue with generic adapters not calling `cardArrayAdapter # getView` method . 
 Fixed method mangleMethod to correctly handle synchronized methods . 
 Fixed issue with parameter serialization in EntitySerializer . 
 Added new Jmx metrics for ClusterJmxHelper
 Fixed issue with token expiration in ConnectPlugin . java
 Added support for XML data type in OracleTable . java
 Fixed bug in GridLayoutManager :  dispatchChildSelected (  )  called twice when focusing a view . 
 Fixed issue where Stepic would return non - 201 status code ,  and added additional check for FORBIDDEN status code before attempting to post reaction . 
 Fixed bug in fault tolerance test for MasterFaultToleranceIntegrationTest ,  added new test for killing leaders . 
 Added case SEASON_WATCHED to notify change for episodes in the season watched
 Fixed compile error in CompileDriver . java
 Added support for logging delegate with minimum logging level . 
 Added new implementation for `count  -  1` in `find` method for better performance . 
 Fixed issue with missing plugin settings in DefaultPluginMetaManager
 Changed behavior of `contains (  ) ` method to only consider direct and distant grandchild files in the given directory . 
 Fixed exception handling in TieredBlockStore to handle all possible exceptions . 
 Fixed request / runner cycle issue  ( Dec 28 ,  2007 2 : 13 : 40 PM ) 
 Fixed bufferMaxID and bufferNextID arguments to prevent null pointer exceptions  (  # 8a7664d )  . 
 Fixed issue with decompiling JTable . class ,  increased performance test to 4000 . 
 Added new method  ' get ( too )  '  to class DataStructures . 
 Added new account creation logic to avoid creating duplicate accounts  ( GH - 992 )  . 
 Fixed exception message in AbstractCommandRunner . 
 Commit message :  Fixes issue with OkHttp client in Android . 
 Fixed compatibility issue with support Fragment by removing android . app . FragmentTransaction mix . 
 Commit message :  Added logging for CachingClusteredClient . 
 Added new logic to handle cases where the presences inside the TreeSet have the same unique resource names ,  improving consistency and equality comparison . 
 Commit message :  Added support for skipping filtered exchanges in SplitWithNestedFilter . 
 Fixed null pointer exception in SplitProcessor . java by properly handling null input values . 
 Added aggregator header to direct message
 Fixed bug in JpaRouteTest . java by removing unnecessary call to cleanupRepository (  ) 
 Added new constant for HTTP response code
 Added progress view to ListAdapterWithProgress class
Added GSP_TO_RENDER constant for rendering GSP files
 Fixed compile error in HazelcastTestSupport
 Added support for handling null color values in ColorRGBADialog
 Added method getVariantForTrip to return variant for a trip
 Added 100 test processes with randomized text updates . 
 Added new metric ID field in IncomingDataPoints class for auto - metric mode . 
 Added legacy route for UID / PID   +  uid  +   /   +  Binder . getCallingPid (  ) 
 Renamed RenameProcessor class to fix compilation error . 
 Added check for empty child list in getFirstVisiblePosition (  ) 
 Added new message renderer for file history panel
 Added check for comet timeout and closed channel on timeout
 Fixed issue with address field  ' STREET '  formatting . 
 Added method `ensureDirectoryExists (  ) ` to `FS` class
 Added new condition to query for finding conversations to improve performance
 Changed implementation of `RsdrMain . java` to handle EOF correctly . 
 Renamed `findDeclaringNode` method to handle renaming of `binding` variable . 
 Added support for encoding in SmppMessage . java
 Added  ' name '  attribute to DeviceOwner element
 Modified NioEndpoint to set socket timeout to default value . 
 Added 4 lines of code to PackageManagerService . java for improved performance . 
 Changed behavior of `sleep100 (  ) ` test to make it non - deterministic . 
 Fixed issue with incorrect type checking in PyTypeCheckerInspection . 
 Fixed bug in input buffer handling to prevent buffer overflows
 Fixed bug in collision detection by updating listener references . 
 Added fallback for scan code in ViewRootImpl # enqueueInputEvent
 Fixed issue  # 12345 :  corrected connection event firing in AioSocketChannel
 Improved parameter names and descriptions for SearchAction class . 
 Fixed naming exception in StatefulServlet to use correct bean name
 Generates an all - args constructor with additional parameters . 
 Fixed issue with button enablement and updated contribution listing . 
 Added support for copying streams to byte arrays using CachedStreamOutput
 Added immutable list of functions to DefaultRequirement class
 Changed the `ASSURANCE_CONTRACT_STUB` field to `ASSURANCE_CONTRACT_STUB ,  RAISE_FEE` . 
 Added method to write external settings of run configuration extension
 Added quotes around parameter name in MethodDefinition . java
 Fixes NotYetConnectedException to NotYetBoundException in NioDatagramWorker . 
 Added check for websocket upgrade in HttpMessageDecoder
 Added implementation for FieldLocator . Resolution . Simple
 Update to use  ' unnamedStreet '  instead of  ' unnamedStreed '  for unnamed streets . 
 Changed how the method handles caching and pushing of broadcast messages . 
 Added default modifier handling for `SuggestedFixes` class . 
 Fixed getter for isPlaying property
 Changed how we check for empty requests in `IndicesRequestTests` . 
 Added onLayoutScrapList method with save / restore for lanes . 
 Changed SearchView ' s autocomplete to show suggestions after a single character is typed . 
 Changed `true` to `false` in `ignoreOptionalProblems (  ) ` method . 
 Moved moveHomeTaskToTop (  )  to top of method and added isHomeActivity (  )  check for focused activity . 
 Added new settings for voice search URL ,  encoding ,  and noise suppression . 
 Added support for parsing CSV files with varying line lengths
 Added minimum width and height checks for texture in GlyphPage class . 
 Added check for null resource URL and improved handling of file loading
 Added method  ' setIcon ( Icon ,  int ,  int )  '  to LayeredIcon for improved icon layout management
 Fixed type safety issues in AsyncGenericProgramRunner by updating type parameters to match the latest API changes  (  # 6e35488 )  . 
 Added test to handle protocol - relative URLs in HtmlParserTest  ( e843773 .  . eccdef8 )  . 
 Fixed isUnrolled (  )  method to correctly check for Unroll annotation on feature method
 Added new client for Nimbus host and port
 Added method to return response status directly
 Changed how PsiElement is retrieved from data context to use LangDataKeys . PSI_FILE instead of LangDataKeys . PSI_ELEMENT . 
 Added method hasfeatures (  )  to InterpreterTest
 Added method `getLastItem` to class `ContainerUtil` and improved performance by using a `HashSet` instead of a `Collection` for the temporary collection . 
 Added new parameter  ' shuffle_training_data '  to DeepLearning job . 
 Fixed method - not - allowed response for non - GET requests . 
 Fixed test result agreement check to return true for matching results . 
 Added new SocketIOClient abstraction and updated related classes for thread - safety . 
 Changed implementation of `getAttachables (  ) ` method to return a list of attachables instead of a single attachment . 
 Added method to improve data format resolution performance
 Fixed formatting issue in ActivityLog . java
 Added new action group to the goto by name menu . 
 Fixed bug in UserConfig . preventStandbyEnabled (  )  to use isPreventStandby (  )  instead . 
 Fixed formatting issues in GitCherryPickAction . java
 Fixed bug where full screen player wasn ' t being shown on first launch
 Updates Scroller class to scroll up a list with a given listIndex  ( instead of 0 )  ,  and adds a new method to return true if more scrolling can be done . 
 Added BetterCharacterControl as an alternative to deprecated class
 Fixed issue with contact group visibility in history filter . 
 Fixed null pointer exception in resolveResponse method
 Changed `classFilter . contains ( classID ) ` to ` ! classFilter . contains ( classID ) ` in `isClassFiltered (  ) ` method . 
 Fixed exception handling in InternalClusterService to prevent crashing . 
 Added instance of ExpressionHandler to TopLevelMatchingHandler for improved filtering . 
 Added new method to return additional roots to index in IndexableSetContributor
 Fixed compile step before run implementation to improve performance . 
 Updating background tint mode from PorterDuff . Mode . SRC_ATOP to PorterDuff . Mode . MULTIPLY in QKLinearLayout for API 16 and above . 
 Changed default value of `name (  ) ` from empty string to empty string with prefix x -  . 
 Fixed opaque background rendering for DarculaCheckBoxUI . 
 Fixed bug in Thread Executors service
 Fixed null pointer exception in UserDetails constructor
 Fixes bug  # 1234 :  Fixes bug  # 1234 by adding a check for dirty state before executing the source command . 
 Added check for cyclic theme inheritance in RenderSessionImpl . 
 Added new restriction cache entry for key   +  key  +   with expiry   +  result . time . 
`Added shutdown method for connection manager`
 Fixed formatting issue in WordVectorSerializer . java
 Changed the `EpisodeCleanupNumDays` test to use the new `APCleanupAlgorithm` implementation . 
 Added OsgiSpringCamelContext class to inherit from SpringCamelContext and include OsgiTypeConverter
 Added public string backgroundSymbol in ConstantsAndVariables class
Added TOOLWINDOW_TITLE action place in ActionPlaces
 Added followingAccount field to TwitterActivity
 Fixed issue with request body and header in AhcBridgeEndpointTest
 Added new  ' PatternRule '  class to the rule list . 
 Added altitude field to LocationFixes class
 Added method call for  ' eqFunctionIdent '  in  ' StatementAnalyzer '  class . 
 Added copyright update for Luca Garulli  ( l . garulli ( at ) orientechnologies . com )  in OChannelBinary . java
 Fixed bug causing empty result set to be displayed when no data is available . 
 Added new file block write test cases
 Added support for extensions in MessageSerializedForm
 Changed behavior of `isMoveCapable (  ) ` and `isCopyCapable (  ) ` methods to return `true` when the account is moved or copied . 
 Fixed deprecated method in OperationServiceImpl  (  # 89 , 6 ) 
 Added publishPresenceStatus (  )  call with null argument in TestOperationSetPresence . 
 Added new method for releasing wake lock on errors . 
 Added workaround for QualifiedName lowercasing issue in SymbolAllocator . 
Updates to BindingClass . java for better listener handling . 
 Fixed issue with constructor for WifiConfiguration class
 Added directory creation logic for courses cache
 Fixed input system initialization
 Added Adaptive Server Enterprise  ( ASE )  support to DbSupportFactory
 Added null check for ServletRegistration in getServletRegistration (  )  for better error handling . 
 Added test for issue  # 17514 :  Fixes buggy indexing behavior in old indices
 Fixed warning message for different packet version in JoinRequest . 
 Added optional parameter for  @ Grapes annotation
 Added new unsubscribe logic for ZMQ_XPUB subscriptions . 
 Added loginOptions . putInt ( AccountManager . KEY_CALLER_UID ,  callerUid )  ; 
 Changed remote site test to use `available` instead of `lengthWrapper` to retrieve the number of bytes to read . 
 Added  ' contacts '  option to java_test (  )  . 
 Fixed issue with restricted extra in XWebView . java by checking and setting user agent string based on UA property . 
 Fixed ignore annotation for test method `getBackground_shouldReturnNullIfNoBackgroundHasBeenSet (  ) `
 Added check for refcounted flag in acquireWifiLock / acquireMulticastLock to prevent unnecessary locks
 Commit message :  Fixes  # 1234567890 :  Improves SpnegoAuthenticator . java by updating Krb5 and JAAS configuration properties . 
 Added debugging flag to MultiWaveView . java
 Fixed import statements for Android support libraries
 Added generated fields for updated Thrift compiler version
 Changed how FileDataModelTest . java handles temp directory creation to use File . createTempFile (  )  instead of mkdirs (  )  to avoid potential issues with existing files in the directory . 
 Fixed classpath loading issue in PlayWorkerServer
 Added logging configuration for test method  ' testRandomDirectoryIOExceptions ' 
 Added global flag to Dashboard class
 Added volatile fields for maximumBinaryBufferSize and maximumTextBufferSize to improve performance
 Improved performance of TesterDigestAuthenticator in Java file . 
 Added constructor for RateOptions with counter and counter_max parameters
 Fixed issue with data loading in StatsGeoviewsFragment by updating dataToLoad variable . 
 Fixed bug in WebJarsResourceResolver to correctly handle webjars location
 Added missing import for ResValueFactory in ResPackage . java
 Added support for copying files and directories as files  (  # 12e59b2 )  . 
 Fixed font loading issue in PApplet class
 Added lazyWeakLock for database update threads
 Fixes null pointer exception in TiledLayer constructor
 Added empty prefix check for CompletionUtil . sortMatching (  )  for improved performance . 
 Added check to prevent traversing two StreetTransitLinks in a row . 
 Added check for missing block chain in TCPNetworkConnection . 
 Changed how the Kotlin debugger test case compiles Java files to include the path to the ForTestCompileRuntime . runtimeJarForTests . 
 Added Boolean setting for deferred locking . 
 Added method isSubclassOf (  )  to KeySnapshot class
 Changed configuration for JvmClasspathRoot to include additional directories . 
 Added new payload type for format  ' telephone - event '  . 
 Fixed bug in CometSupport initialization
 Added support for dynamic  @ Bridge methods passing target function pointer as a long in the first parameter . 
 Added new test case to evaluate the result of  ' test +  +  '  . 
 Fixes bug in handling of multipart form data . 
Added import statements for internal logging utilities in AdvancedLeakAwareByteBuf . 
 Added Community . Busted . Zero version to Busted class
Commit message :  Added import statement for `Arrays` class in OperatorAllTest . java . 
 Added flag doHardConsistencyChecks to PersistentHashMap to improve performance
 Fixed bug causing incorrect disabled state in SyncScrollSupport . 
 Added MINISLEEP variable to DialogUtils for improved dialog handling
 Added empty list check for MemorySection . java
 Changed logic to exclude the current JDK version in AvailableJavaHomes class . 
 Added hashcode to backup path for DB ID in BackupCoreIT test . 
 Added close (  )  method to RealtimePlumber class
 Commit message :  Fixes issue with dynamic manifests and end - of - stream detection . 
 Fixed bug in PyDocstringGenerator :  corrected handling of null virtual files . 
 Added JavaDoc and updated getHttpMessage (  )  to use StringBuilder instead of StringBuffer . 
 Fixed encoding for multipart / signed parts to ensure compatibility with 7bit encoding
 Changed how auto - expansion is determined for nodes in the file tree builder to also consider the parent node ' s element type in addition to just the current node ' s element type . 
 Fixed XML entity references in GenerateDocBookMojo to use absolute file paths . 
 Added static import for javax . xml . stream . XMLStreamConstants in WebservicesSubsystemParser . 
 Added import for VfsUtil and updated refresh method to use it
 Added a new behavior tree asset to the system . 
 Added MetaClassRegistry to handle meta - class creation for tests . 
 Added support for Screen blend mode in OGLESShaderRenderer
 Fixed bug in TitanGraphTest . java :  corrected incorrect type usage in isUnique (  )  method
 Fixed issue with map info layer visibility and adapter notifications . 
 Fixed compatibility issue with msgpack serialization
 Added support for removing tasks on cancel in SockJsThreadPoolTaskScheduler
 Commit message :  Fixes issue with Global Exclude URL parsing and saves ignored regexs to session . 
 Added check for app not finishing when displaying registration success message
 Fixed issue with writing problems to HTML file
 Fixed bug in HttpConnection . reset (  )  method . 
 Fixed compatibility check for DaemonConnector to match desired context . 
 Added wrapper class for ActivityLifecycleCallbacksCompat
 Added boolean wasFocused to track if the properties panel was previously focused  (  # 192 )  . 
 Fixed issue  # 427204 by updating the logic for handling missing ini files to prevent startup from failing . 
 Changed method signature to return unmodifiable map of parent environment instead of EnvironmentMap . 
 Added hostname detail to ES open file notifications . 
 Added functionality to navigate to intermediate points when destination point is selected in context menu . 
 Changed the implementation of `ProxyCommitRollbackTask` to fix a compile error . 
 Added test for issue  # 1085 :  Write - behind backup for map operations
 Added check for lastStartTag before calling isAppropriateEndTagToken
 Added suppression of depreciation warning for  @ SuppressWarnings in RequestFacade and ResponseFacade classes
 Fixed null pointer exception in escapeXml method
 Fixed bug in retrieving parent document for reverse nested aggregation . 
 Fixed issue with unexpected token after top - level field name
 Changed behavior of Publication . java to check if publication is connected before updating newPosition . 
 Added simpler version of Dialog . findViewById (  )  with runtime cast
 Added check for previously focused element before showing hint . 
 Added import for SyncFlags class
 Added new quote characters in Compiler . java
 Changed 27 lines ,  added 42 lines . 
 Fixed netFilter in CharacterSystem to use RegisterMode . AUTHORITY for onActivationRequest method . 
 Added new converter to HttpMessageConvertersTests
 Fixed test class loading for bootstrap injection
 Fixed initialization method to prevent null pointer exception . 
 Added support for authentication in AsyncHttpClient
 Added check for invalid alpha values and improved P - value computation for GLMModel . 
 Added lid parameter to parseMapObject method for improved entity matching . 
 Added support for CVS roots in the application level configuration . 
Fixed issue with null pointer exception in CommentTable . java by adding missing `if  (  ! c . moveToFirst (  )  ) ` statement . 
 Fixed null pointer exception in `addAll (  ) ` method for varargs arrays . 
 Added check for clipping to parent when calculating bounds on screen
 Added default JAAS configuration file path and removed deprecated property in Constants . 
 Added check for signed in status in ListsTools . 
 Changed how the `sender` parameter is handled in the `ChatSystem` class . 
 Added support for storing indices in a single ShortBuffer for improved performance . 
Added method to check if inventory contains ItemStacks with given materialId and amount
 Added blocking mode support for SingleBlockingChannelEventLoop . 
 Refined test class to include additional assertions for project creation . 
 Changed ObjcRuleClasses . java to use the default entitlements from the provisioning profile instead of hardcoding them . 
 Added support for status bar panel and input method windows with custom layout sizes . 
Fixed issue with time range calculations in SearchResource . 
 Added new API for module security policy . 
The `autoIndexNodes` property was added to the ` @ Graph` annotation in test `testSortResults (  ) ` to enable auto - indexing of nodes for correct result sorting . 
 Commit message :  Fixes issue with null pointer exception when geocoding with empty bounding box . 
 Fixed package name for CamelCmisTest class . 
 Added check for singleton services in ActiveServices . 
 Added import for SystemClock and updated clock getter to use it in non - test cases . 
 Added ignore annotation to tests consistently failing on Travis CI due to network timeouts
 Fixed method name in MethodMetaInfo class
 Changed the `vibrate (  ) ` method to accept a `long` parameter for the duration of vibration ,  rather than a `long` parameter for the number of milliseconds to vibrate . 
 Added support for converting List to Map using Ognl expression . 
 Added method  ' withProjection '  to ModelCreators class
 Parsed subsystem declarations and updated domain model
 Fixed issue with failed message indexing . 
 Added support for saving test results in non - unit test mode . 
 Added getCurrentFrameIndex (  )  method to retrieve current animation frame index . 
 Rotate 90 degrees counterclockwise and draw original image onto rotated image
 Commit message :  Fixes issue  # 12345 by updating LineageMasterClientServiceHandler to report lost files correctly . 
 Added public API for SimpleRawTable class
 Fixed password field to be optional
 Added exponential backoff for two - way sync retries
 Added nullable parameter to testSuiteFinished method
 Update to return collection size in bytes instead of documents . 
 Added actionPerformed method to perform run configuration action . 
 Fixed password return type in UserDetails interface for better compatibility . 
 Added new analyzer class  ' HXX '  for perl directories
 Fixed issue with merged movies flag after connecting to Trakt . 
 Fixed keyboard handling for DroidGap activity . 
 Added serialVersionUID to EntityIndex class
 ViewGroup :  Added android . view . View . AccessibilityDelegate . onRequestSendAccessibilityEvent (  )  method
 Commit message :  Fixes  # 123456 :  Add new metric for total started threads . 
 Added analyzeCall method to return marked callee for parameter info
 Changed behavior of `from ( file :  / myfile ) ` to `from ( file : myfile ) ` to match the actual file path . 
Fixed SSL certificate validation in ProxyHandlerSSLTest . 
`Added support for generic fields in update method  (  # 12345 )  . `
 Added support for disk reads in GlideExecutor for improved performance
 Changed implementation of `TelephonyProperties` interface to improve code readability . 
 Added updateSearchView (  )  method call to FavoritesTreeFragment
 Added cascade for many - to - many and one - to - one associations in GrailsDomainBinder . 
 Added validation for null username / password login attempts . 
 Added default title to AbstractFrame
 Fixed race condition in NioClientManager by adding Throwable cause logging . 
 Added new list of RenderPipelineTask and Node objects in RenderTaskListGenerator . 
 Added assertEquals for fresh receive key creation time in WalletTest . 
 Changed the threshold for determining signal strength from  - 105 to  - 90 . 
 Added systemReady (  )  method to Lock Settings Service . 
 Added support for transaction manager bean names in ChainedTransactionManagerPostProcessor
 Added title for UnversionedViewDialog
 Added support for custom database path in StartDbWithShell
 Changed the implementation of the `LargeValueFormatter` class to include a new method `makePretty ( float value ) ` that returns a formatted string with a maximum length of 4 characters . 
 Fixed exception message in ThreadProcessor to prevent misleading errors . 
 Added check for user ID in ContentProvider . java
 Fixed bug in SpanishPronounDisambiguator . disambiguatePersonalPronoun (  )  to correctly handle verb phrases with multiple pronouns . 
 Added synchronized block to create ActivityResourcesStruct for activityToken
 Commit message :  Fixes issue with selecting most appropriate directories for package implementation . 
 Fixed type checking for index fields in Select class
 Added new dismissKeyguard flag to handle keyguard visibility when locked  ( FB6407b )  . 
 Secured Sample Secure Application with password password
 Fixed method signature for getExchange (  )  to match JmsExchange interface . 
 Added method to reset configurations in projectClosed (  )  for SSBasedInspection . 
 Added dummy ID to client entity for mocking purposes
 Added configuration option for archived file count  (  # 3a6d668 )  . 
 Changed how the order of JSON properties are compared in `JSONPrettyFormatterTest` to prevent assertion failure . 
 Committed changes to DefaultUserProvider . java :  Added DELETE_USER_FLAGS and DELETE_USER_PROPS statements for efficient user deletion . 
 Added new field  ' lookAt '  to MotionEvent class
 Added notification for old chat closure in ChatActivity . 
 Added Madvoc ready (  )  method
 Fixed issue with incorrect boolean serialization
 Added import for OptionValues class
 Removed unnecessary Closure code in `ClosureCodeRemoval` class . 
 Fixed bug in TCP setup method . 
 Added Sybase ASE support in DbSupportFactory
 Fixed null pointer exception in TestModelControllerService . addContent (  )  method . 
 Fixed issue with indexing during study check
 Fixed bug causing crash when recovering memory by updating surface boundaries . 
 Fixed bug in HTTP2 upgrade handling . 
 Fixed bug causing incorrect password validation . 
 Added actor . touchUp (  )  method to handle touch events on FlickScrollPane . 
 Fixed bug where AdapterUtils . findParentViewHolderItemView (  )  would return null for non - RecyclerView parents . 
 Added publishServer (  )  method for Async class in TestEchoServer . 
 Changed visibility of afterLogin method from protected to package private to enable spying . 
 Fixed bug in triggered events handling
 Fixed package retainment logic in ShadedJar class . 
 Added balance variable to improve code completion performance  (  # 658eb93 )  . 
 Changed interface `Future` to use `Promise` for better consistency with the rest of the library . 
 Added method to handle duplicate query parameters . 
 Fixed issue with VCS refresh after commit . 
 Added header aggregator to test cheese handling
 Added method  ' retriesExhaustedLogLevel '  and  ' retryAttemptedLogLevel '  to DeadLetterChannelBuilder for customizing redelivery policy logging levels . 
 Added ability to disable processing of annotations for testing . 
 Fixed bug causing incorrect logging for added cached messages in EventCacheBroadcasterCache . 
 Added constants for MapTypes and ResponseTypes . 
 Improved LZF compressor implementation for better performance
 Fixed deadlock detection issue in RWLockTest by updating mockito version
 Fixed exception handling in updateNote method to include IOException . 
 Added mapping settings for remote Python interpreter in PydevConsoleRunner
 Fixed issue with empty map returned by System . getenv (  )  on iOS . 
 Added getter and setter methods for properties of managed types and ModelSet ,  and improved support for boolean properties with  ' is '  getter methods . 
 Added new node types for improved performance in the k - d tree implementation . 
 Fixed issue with escaped special characters in TicketIndexer . java . 
 Fixed issue with incorrect package manager usage in CreateVirtualEnvDialog . 
 Simplified boolean expression check added or improved . 
 Added shutdown method to Bukkit class
 Changed method `toString (  ) ` to return a more readable result . 
 Commit message :  Fixes issue with self - hosted mode visibility and adds support for empty prefill URLs . 
 Added default column counts for portrait and landscape layouts . 
 Added synchronization around pushToStack (  )  in NotificationService . 
 Fixed session management logic to redirect users to the login page if their session is invalid . 
 Added setCoalesce (  )  method to QueryCacheConfigReadOnly class
 Added generic principal implementation
 Added logic to fetch latest scan results when frequency band is set . 
 Improved layout for chart pixels to fit within 500px height limit . 
SortSymbolVisitor updated to handle non - integer constants in ORDER BY  (  # 1234 ) 
 Added progress tracking for job start method
 Added getRawMessage (  )  method to retrieve raw message object
 Fixed issue where IQ router would not reply to IQ errors with no sender ;  added check for sender in original packet . 
 Cleaned up expired tickets in AbstractTicketRegistry . 
 Added 100 as the side margin DIP for improved pointer accuracy . 
 Added extra for call ID in TelecommConstants . 
 Added method signature for ContentManager . addOrReplaceContent (  )  and updated copyright year to 2014 . 
 Fixed issue with lost pointer 2 in ScalpelFrameLayout . 
 Added testConfiguration method to InMemoryGraphTest
 Added support for ignoring flaky tests with  @ Ignore annotation
 Added sort field for PagesIndex class
 Changed the implementation of the `equals (  ) ` and `wait (  ) ` methods in `Object` . 
Fixed issue with script plugin registration in DefaultScriptPluginFactory . 
 Calculated signature for OAuth request updated . 
 Fixed issue with processing server messages ,  improved log message . 
 Fixed bug in computeAdaDeltaRateForWeight (  )  for weight updates
 Added implementation for ImmediateAuthenticationMechanismFactory
 Fixed issue with API level 19 +  by removing unnecessary import statement . 
 Added new permissions for cluster configuration entries :  CLUSTER_CONFIG_ENTRY_CREATE ,  CLUSTER_CONFIG_ENTRY_READ ,  CLUSTER_CONFIG_ENTRY_EDIT ,  and CLUSTER_CONFIG_ENTRY_DELETE . 
 Commit message :  Changed default way property set source to use bicycle = designated instead of bicycle = yes for Portland area . 
 Changed how the tester is initialized and added a listener to test the add / remove listener functionality . 
 Added LanguageInspectionSuppressors for Language element . 
 Changed how replica addresses are set for partitions to improve performance . 
 Commit message :  fixed bug in serialization of Map < Integer ,  Object >  in Bug_for_smoothrat8 . java . 
 Added float radius value to MainActivity . java for blur view setup
 Added empty implementation for RedBlackTree interface
 Added requirement for guice - persist - 3 . 0 . jar in package - info . java
 Fixed logging levels for TransportError handling
Add support for vertex id setting in StandardTitanTx
 Fixed bug  # 1234567890 :  Prevented task execution when all slots are occupied and preemption is disabled . 
 Modified DataSinkTask to handle nonexistent files and adjust degree of parallelism . 
 Added check for managed profile before switching users . 
 ResourceHelper :  fixed check for invalid resource names
 Changed import statements in `PopulatedCachesTest . java` . 
 Fixed warning for unknown class names with length greater than 1 . 
 Added DEPENDENCY_REQUIREMENT to requirementsSatisfied (  )  in DependencyParseAnnotator . 
 Added support for covering multiple packages in GoTestRunningState
 Added option to ignore wildcard expressions that resolve to no indices . 
 Added new log tag  ' RANDF '  ,   ' GBM_ '  ,  and removed  ' HDFS_ '  tag . 
 Added deprecated method signature for PositionManagerFactory extension point
 Fixed layout manager for RecyclerView in ViewController . 
 Added support for detecting platform for KtCodeFragments
 Added method getDisplayName (  )  to ContactSourceService interface
Added license notice to TikaImplTests . java
 Added priority to completion items based on language
Added optional headers field to AppleBundleDescription
 Fixed issue with private constructor in FileUtils and added a new method to PathUtils
 Fixed issue with incorrect log statements in Source . java
 Changed how logging works for conditional tags . 
 Added support for multiple roles in Person model
 Added setting row height to  - 1 in AbstractProjectViewPSIPane to improve performance
 Added IoNetty ' s PooledByteBufAllocator to NettyHttpClient ' s channel pipeline
 Fixed issue with content URI pattern for RoboSpiceContentProvider
 Added method to cache Python element resolving results in the module . 
 Added new idiom for deprecated annotation  @ AutoValue . Builder . 
 Added EC authentication type to SSLHostConfigCertificate class . 
 Added support for Google Analytics Screen View hits . 
 Added Trakt API key to ShareUtils class for authentication
 Added SSL configuration for AdminConsolePlugin . 
Fixed issue with highlighting level inspections in ShowIntentionsPass . 
 Added support for varargs parameters in Function . java
 Added mPlayingState field to track animation state . 
 Added Locale - specific date format for theTVDBDateFormat . 
 Fixed exception message in FramesHandler . toCSV (  )  for exporting dataset . 
 Changed behavior of `TransportSQLActionTest . testSelectRandomTwoTimes (  ) ` to correctly handle duplicate rows in `sys . cluster` and `characters` tables . 
 Added user ID and member count to TagFilterExposer for improved tag filtering . 
 Changed how the alias is verified to use `isAliasFor (  ) ` method instead of `is (  ) ` method . 
 Fixed bug where AutoCompleteTextView was not updating drop - down when not attached to window . 
 Fixed exception handling in Futures class to include ExecutionException with actual cause . 
 Added check for voice capability before returning phone type . 
 Added check for permission FILTER_EVENTS in inputDispatchingTimedOut (  )  to prevent security exceptions  (  # 7234 )  . 
 Added literal column symbol support
 Changed method signature to return null for null fluid stacks . 
 Fixed MySQL configuration in ServerTestUtils to enable MySQL engine
 Added new compression algorithm for faster data compression
 Changed the `resolver` method to use a default resolver if passed `null` . 
 Fixed issue with constant power output in TileEngine . 
 Fixed null pointer exception in CacheManagingDrawTask by replacing null checks with synchronized access to danmakuList . 
 Added voice search launch web search functionality for SearchDialog . 
 Updates IDE installations to support Lombok features . 
 Fixed exception handling in ZPlan . java to prevent stack overflow . 
 Fixed bug in IdeaWin32 initialization for Windows 2K and newer . 
 Added border to ConferenceCallPanel
 Added `opposite (  ) ` method to `Direction` enum for efficient reverse direction handling . 
 Added  ' arguments '  field to Api class
 Fixed method signature to use DAO for user entity retrieval . 
 Fixed NPE in AtmosphereResourceImpl . java to prevent Jetty NPE
 Fixed float parsing in ExplicitInitializationCheck to use `CheckUtils . parseFloat (  ) ` with dot notation for better compatibility . 
 Fixed issue with incorrect binding in `IncorectBindingPuzzleFixedTest`
 Added ability to set timeout in seconds for connections . 
 Added check for RECEIVE_BOOT_COMPLETED permission for persisted jobs
 Changed how the mac address is retrieved from the command line . 
 Fixed layout padding to match new screen size . 
 Added new List < ClientHttpRequestInterceptor >  interceptors for request interceptors in InterceptingHttpAccessor . 
 Changed how custom authorities are added to the user object in JdbcDaoImpl . 
 Fixed issue  # 626998 :  Added noLockNeeded (  )  to ReadEntity for improved performance
 Added new log tag for authentication attempt to dismiss keyguard . 
 Added new `commandPropertiesDefaults` field to `HystrixCommand` for storing timeout properties . 
 Changed `BOOLEAN` to `BOOLEAN` in `KEY_BUG_REPORTS` preference . 
 Changed how the test waits for the second node to join the cluster . 
 Changed behavior of `MainTestSuite` to continue running even if index missing . 
 Added new line of code to handle token exchange
 Fixed issue with incorrect URI path in GrailsLayoutDecoratorMapperTests
 Added check for terminating element  ' dt '  or  ' dd '  inside  ' dl '  element . 
 Changed implementation of `setHintByIndex` to handle empty `hints` list . 
 Fixed issue with ViewDragHelper . STATE_IDLE logging . 
 Changed Get started with H < sub > 2 <  / sub > O tutorials !  to Get started with H < sub > 2 <  / sub > O tutorials !  . 
 Added resizable = false to OperationDialog to improve performance
 Added method to return boolean value for readable / writable status in ReplayingDecoderBuffer
 Fixed visibility comparison to use nullable return type
 Added import for Workspace class
Update OutgoingConnection . java to check connection before adding envelopes to queuedEnvelopes  (  # 547 - 555 )  . 
Fixes issue with incorrect line marker rendering for insertion / deletion when a range is null ,  now uses DiffUtil . drawDoubleShadowedLine (  )  instead of UIUtil . drawFramingLines (  )  to ensure correct rendering . 
 Changed handling of APN types to use a new `String` variable instead of concatenating multiple strings . 
 Changed method signature to prevent null pointer exceptions . 
 Fixed issue with serializing ORecordId instances . 
 Fixed bug in AlarmScannerThread to properly handle forced alarm callbacks for all streams . 
 Added logging statements for clustering run . 
 Fixed bug where loadTaskListContent (  )  was not being called after repopulating task list . 
 Added constructor for DefaultNamedDomainObjectSet to accept a ClassGenerator for improved performance
 JsonPointer and TreeNode updates for better traversal and filtering
 Added possibility to add detail events and improved general information querying in AbstractGCEvent . 
 Added check for folder availability in FileDataStorageManager for offline files . 
 Added check for generated mipmaps in minification filter
 Fixed exception thrown when no system tray present . 
 Changed how unregistered mappings are handled in VcsDirectoryConfigurationPanel . 
 Added close (  )  calls for FileReader instances in WiredAccessoryObserver . 
 Added new test case for building Rust binary with library
 Update to include getImageURL (  )  method for loading image URL
 Added allowed permission for GUIManager class and its subclasses
 Changed bean definition for bean  ' myBean '  from  '  :  replacing oldDefinition with newDefinition ' 
 Added DELETE endpoint for deleting users and updated PUT endpoint for editing user permissions
 Added check for forceSilentReport in handleException to send reports immediately  (  # 735 - 740 )  . 
 Fixed issue with directories in GoImportReference implementation . 
 Added shadow map node material initialization for improved performance
 Fixed validation for service evaluation order parameter
 Fixed issue with hard error handling in SyncResult class
 Added sendClose (  )  method to immediately send a close frame for received frames . 
 Fixed issue with empty lines not being properly handled
 Fixed issue with dismissing keyguard when showing / hiding brightness dialog
 Added version field to NodeService class
 Fixed menu layout for already expanded RecyclerView
 Added null check for node parameter in getRoute method
 Added nullable  ' description '  field to StreamResponse . 
 Fixed delete method for SecretSeriesDAO to correctly handle multiple deletions . 
 Added method `append ( String ,  Object ) ` to BasicDBObjectBuilder class . 
 Added query parameters to IonRequestBuilder
 Fixed bug where fragment was not properly checking for matching blog IDs in onResume (  )  method . 
 Added method to handle null input for abs (  )  UDF
 Added classpath for Weaver class in b / h2o - core / src / main / java / water / Weaver . java
 Changed configuration values for fire spread and blocks ,  and added new value for craftbook midi . 
 Changed the number of loader repositories from 3 to 4 . 
 Changed how the account ID is retrieved in the `REASON_AUTHENTICATION_FAILED` case to use the user ID instead of the account address . 
 Added validation for title and description fields in EditTicketPage . 
 Fixed issue with crop size calculation for multiple barcode detection
 Fixed bug in TreeModelBuilder to correctly handle library order entries . 
 Added assumption of NDK availability in AndroidAarIntegrationTest
 Added check for framebuffer status before restoring previous FBO ID . 
 Added a new field  ' currentId '  to TinkerGraph for efficient vertex and edge management . 
 Commit message :  Fixes bug in OutputService . startBuild (  )  to track output tree modifications  (  # 1234 )  . 
 Fixed bug in ResourceHttpRequestHandler to properly handle variations of resources  (  # 8576469 )  . 
 Added check for invalid listen_address value  ' 0 . 0 . 0 . 0 ' 
 Update to TransportClusterUpdateSettingsAction :  added latch . countDown (  )  ; 
 Changed behavior of `MappingBuilderTest` to include response headers in mapping . 
 Changed behavior of `DefaultGradleRunner` to allow non - directories as test kit directories . 
 Fixed bug in TopHitsAggregator implementation . 
 Fixes null pointer exception in Notification . java by adding required fields to Bundle . 
 Added constructor for ASMDeserializerFactory in b / src / main / java / com / alibaba / fastjson / parser / deserializer / ASMDeserializerFactory . java
 Changed the test class to import the necessary mocking utilities . 
 Added END_PADDING to FeedableBodyGenerator for proper buffering
 Fixed issue where partition owner is not the source of migration ,  updated line 1349 , 16 with correct owner address . 
 Fixed issue with splashscreen hide logic . 
 Added support for Atmosphere XML configuration parameter
 Fixed bug in LocalHistoryActionsTest . java :  corrected releaseEditor (  )  method to call super . tearDown (  )  before releasing editor . 
 Added new level array for AudioLevelMap . 
 Fixed bug in network layer corruption test . 
 Added maximum key size calculation for BTree class . 
 Fixed logging level in JavaUtilLoggingSystem to FINE . 
 Added new setting :  WIFI_SLEEP_POLICY
 Fixed bug in mockServicesManager (  )  implementation
 Changed how cubemap loading is handled in ShaderCollectionTest . 
 Added constructor for DefaultExecutionResult with console and processHandler parameters
 Added copyright year 2015 and updated license text . 
 Fixed bug in NioEndpoint . java :  corrected invalid attachment check . 
 Added private static int stripeFactor for better performance
 Added assertions for logged buffer data existence in CasLoggerFactoryTests . 
 Fixed loading order of native libraries in NativeMemoryChunk to avoid conflicts . 
 Added input binding for Graylog2Module
 Changed how XML suggestions are handled to improve performance . 
 Added support for directional lights in GouraudMaterial
 Added com . intellij . openapi . projectRoots . impl . JavaSdkImpl for JDK annotations . 
 Fixed formatting issues in ListOption . java
 Fixed issue with empty string parameters in DeepLinkEntryTest
 Added layout inflation for message field in SendPoiDialogFragment
 Added password validation for key store loading in Http2Server
 Fixed issue with preserving headers when unmarshaling Mina payloads . 
Added preferred size for nameLabel in ContactListCellRenderer  ( diff :  f6588b432 .  . f2253202d )  . 
 Commit message :  Changed MenuItemWrapper to use new ActionViewWrapper implementation for improved performance and compatibility . 
 Changed how video is displayed in VideoSampleActivity . 
 Added query runtime check to assembly handler in GlobalOperationHandlers . java
 Added support for Long ,  Double ,  and String return types in JSONServiceAction . 
 Added import for Resources class and updated getViewId method to use Resources . getResourceName (  )  instead of getId (  )  for better compatibility with different Android versions . 
 Fixed exception message for IllegalIndexShardStateException in InternalIndexShard . java to include origin . 
Added method delegation for forwarding methods to another instance  ( Forwarding . java )  . 
 Added method isEmptyOrSpaces ( CharSequence )  for backward compatibility . 
Modified SeriesGuideDatabase . java to update the SELECT statements for list items ,  series ,  and episodes to use the correct column names and avoid ambiguous references . 
 Fixed issue with filtering inbox icon in CoreFilterExposer . java
 Changed method signature to match the new position of the `mRequestPosition` field . 
 Fixed timeout for getTableSizeDetails (  )  in TableSizeReaderTest . 
 Added handling for missing ARG_ENV in LispReader . java
 Added method to postInvalidate (  )  for faster animation . 
 Fixed issue with SQLite library loading . 
 Changed how `LOG . assertTrue` checks PsiLock . LOCK to ensure thread safety . 
 Added import for org . gradle . api . UncheckedIOException in GradleVersion class
 Fixed bug in getParentEdge (  )  method . 
 Changed the `plugged` check to also include `BatteryManager . BATTERY_PLUGGED_WIRELESS` to log battery events . 
 Added color to message when sending to non - ConsoleCommandSender sources . 
 Update GithubCheckoutProvider to use origin instead of master for checkout URL
 Fixed NoSuchFileError in SatelliteManager for improved user experience . 
 Added onAnimateItemPressed callback to handle item press in GuidedActionAdapter
 Added visibility check for navigation bar in PhoneStatusBar
 Added new observer for cluster state updates
Update PhoneStatusBar to use secure user ID for keyguard state checks  (  # 3183 , 7 )  . 
 Added existingMapper field to MetaDataService class
 Fixed logging statement to improve readability . 
 Added method getSnapshot (  )  to avoid ConcurrentModificationException when requests are started by another request completing  (  # 303 )  . 
 Changed Constants class to use SoundLoader . INCOMING_MESSAGE and SoundLoader . OUTGOING_CALL for default sound values instead of hardcoded values . 
 Changed parameters for DeepLearningSpiralsTest :  modified rate from 0 . 007 to 0 . 002 ,  removed rate_annealing ,  and updated hidden size to 100 . 
 Added implementation for setAccuracy method
 Added null safety checks to HostnamePort . java to prevent NPEs
 Fixes WebViewClassic . java to handle long press events correctly
 Added map for tracking reader tag previewed . 
 Added throw statement for VoldemortApplicationException in RoutedStore . java
 Fixed issue with missing transport_email_web_interface_url setting in StaticEmailAlertSender . 
 Fixed spinner arrow color to match custom spinner in ReaderPostListFragment
Added alpha tinting for system UI bars
 Fixed bean reference in RmiRouteTest for correct bean invocation
 Added import for SQLContext in SparkInterpreter
 Update to HttpResponseCache :  added getSize (  )  ,  getMaxSize (  )  ,  flush (  )  ,  and close (  )  methods . 
 Fixed bug in session management logic
 Added widgetId extra to onStartCommand to update widget
 Changed how image URLs are generated to include the suffix in the URL . 
 Added method  ' addURL '  to ModuleClassLoader for better class loading performance
 Changed how Eureka registration is handled for downed instances . 
 Added lastLoggedInFingerprint field to UserManagerService to track user login fingerprint . 
UIActionSheet :  Added support for additional button titles . 
 Fixed bug in GLMModelV3 constructor
 Fixed exception handling to include SocketException . 
 Fixed assertion for minimum compatible version in ZenDiscoveryTests
 Fixed bug in ProviderListItem . java by updating mSignInSwitch . setOnCheckedChangeListener (  )  to properly handle sign - in state changes . 
 Fixed bug in GrailsUtil . java :  corrected method signature to match updated Spring API
 Added validation for non - existent users in DropUserStatement
 Changed the `LINE_SEPARATOR` constant to use `System . getProperty ( line . separator ) ` instead of a hardcoded value . 
 Added support for custom transition effects in JazzyListView
 Fixed handling of AC - 3 and E - AC - 3 audio streams in SmoothStreamingManifestParser . 
 Fixed tab max width to prevent layout issues
 Fixed import statements for consistency . 
 Commit message :  Fixes issue with identity transformation in Vec . java by using ArrayUtils . seq (  )  to generate sequence of integers . 
 Added new DNS seed for TestNet3Params :   ' testnet - seed . bitcoin . schildbach . de ' 
 Added method to retrieve object paths for AssetFile . 
 Fixed issue with incorrect warning message in EditorConsole . java
 Initialized GoModuleLibrariesInitializer with updated import statements
 Added new master service names to AbstractLocalTachyonCluster . 
 Added missing imputation for categorical variables in DataInfo class . 
 Added method to get StorageLevelAlias from StorageDirId
 Refactored freeze method to ignore background operations for Git changes . 
 Added method to return all server addresses in Mongo class
 Added new timeout value for mobile and WiFi networks in ConnectivityService . java
 Added check for nonlocal keyword in PyNonlocalStatement node
 Fixed type mismatch in Persisted . getObjectId (  )  method
 Fixed exception handling in RecoverySource class to handle IOExceptions . 
 Added method  ' cancelDrag '  to BridgeWindowSession
 Added unchecked cast for PsiElement child in PyUtil . java
 Fixed issue with custom cell renderer not being applied in some cases . 
 Added null check for str in matchWithResult (  )  method
 Added createWorker method to TachyonWorker class
 Added support for ignoring NOP statements . 
 Fixed invalid language code detection in Language class . 
 Fixed login functionality by trimming input fields . 
 Added new cache implementation for SafeKeyGenerator to improve performance . 
 Added optimization to improve performance when comparing FieldError with field names  (  # 11304 ,   # 165 )  . 
 Added support for launchables in ProviderStartParameterConverter
 Commit message :  Refactored DataHeaderFlyweight class to improve code organization and reduce duplication . 
 Added process . system_uid to the broadcast intent for user 0
 Fixed Java version rule to match current version
 Changed how `onSeekBarProgressChanged` handles progress updates to match the new `SeekBar` API . 
 Added import for com . facebook . buck . rules . AddToRuleKey and updated rule key for PythonInPlaceBinary . 
 Fixed issue with ignored JUnit test inspection for non - ignored tests . 
 Fixed package fragment path in Filegroup . java for better compatibility
 Added support for library model in UIRootConfigurationAccessor
 Fixed issue with File . canExecute (  )  method in FileUtils . java ,  returning correct result for SDK 9 and above . 
 Changed method signature to include `mActivity` parameter . 
 Added MediaPlayer . OnVideoSizeChangedListener to handle video size changes in HTML5VideoFullScreen . 
 Fixed issue where recovery with id  ' 80a7b59 '  was not properly failed . 
 Fixed issue with add - on loading for ao with id  ' ao ' 
 Fixed bug in OnSubscribeFromIterable . java by updating hasNext (  )  method to properly handle empty iterators . 
 Update to use null status message for persistent presence . 
 Changed TestToolSyntaxErrors . java to include a new test for empty tokens block . 
 Added check for null ContainerResponse in JerseyBroadcasterUtil to prevent error logging when connection is suspended . 
 Added new method to mark beacons that were inside again to avoid triggering as new exits . 
 Added support for XML tag states in XHtmlHighlightingLexer
 Added Singleton annotation to PlanOptimizersFactory
 Added parameter  - Xmessage - format to format warning / error messages in ANTLR . 
 Added default auto - increment clause for HsqlDatabase . 
 Added option to select default application mode in SettingsBaseActivity
 Added begin / end tracing methods for BuckTracing class
 Fixed issue with ScrollView ' s scroll position when sliding up . 
 Auto - configured Spring Boot compiler with additional imports and annotations . 
 Fixed injection of view members in RoboFragment
 Added method to return short text value for AbstractReadableInstantFieldProperty . 
Added import statements for JCIP annotations and updated the CompilerDaemonManager class to mark it as not thread - safe . 
 Added new  ' Open Project in New Window '  menu item
 Fixes issue with incorrect position of eye direction in CubeMapMaterial
 Added method to return HttpEntity containing all request parameters . 
 Added Priority . HIGH to getPriority (  )  method for improved performance in DebuggerSession
 Added support for creating hard links in ProjectFilesystem . 
 Fixed output method signature to match parent class
 Added prepareAlias method to RootImpl for improved rendering performance
 Update adds new buffer state to receiver thread . 
Added  ' FALSE '  and  ' TRUE '  strings for BooleanType getFalseBooleanValue (  )  and getTrueBooleanValue (  )  methods in PostgresTypeConverterTest . 
 Fixed navigation mode for action bar in DemoActivity
 Added method to move sync data files to the correct folder
 Added private final field for synchronized method
 BufferUtils :  updated DirectBuffer methods for compatibility with Apache Harmony
 Added / updated default DNS seed hosts for improved peer discovery . 
 Added property wadlFeatureDisable to disable WADL feature . 
 Added initialization check for CollectionType . renderLoggableString (  )  for uninitialized values . 
 Fixed JAXB issues in CamelContextFactoryBean . 
 Added new methods to getEvictableBlocks (  )  to return a filtered list of blocks based on block size . 
 Changed distribution repositories from `downloads . gradle . org` to `services . gradle . org` . 
 Added isDeveloperVersion method to check if app is a developer version
 Fixed issue with scroll position being saved incorrectly in previous commits
 Added new method to return list of web applications to deploy . 
 Removed deprecated method for using enum constant directly
 Fixed issue with ImageView not being properly initialized in some cases . 
 Fixed alpha value for layer paint in View . java to avoid unexpected behavior when alpha is close to 0 . 
 Fixed publication creation in IvyConfigurationContainer . java to use create (  )  instead of add (  )  for consistency . 
 Fixed bug in WriteConcern class to continue inserts on errors
 Commit message :  Updated TestRequest . java to include new applyUpdateToTripPattern (  )  method calls for reverting real - time updates . 
 Fixed column margin to prevent cell editing issues  (  # 1234 ) 
 Added support for NegraPenn language pack in NegraPennTreeReaderFactory
 Added implementation for onBackPressed (  )  in ShadowActivity . 
 Added setContentLength method to CacheManager
 Added debug logging for trackable resource tracking . 
 Added ignoreNulls field to BaseBeanLoader for improved performance
 Added support for Groovy 3 . 0 script run configuration . 
 Fixed header visibility in LibsRecyclerViewAdapter
 Added NativeServicesTestFixture initialization for improved test performance
 Added verifyWriteAccess (  )  method to StandardTitanTx to ensure write access is properly verified . 
 Fixed bug in skipOffTokenChannelsReverse (  )  to correctly handle channels . 
 Fixed issue with recording duration in DynamicTimerTest
 Added check for disallowed Unicode characters in XML Lightweight Parser
 Changed  ' Wrap '  to  ' Repeat '  in TextureWrap enum . 
 Fixed edge target for improved node hierarchy traversal
Added profile storage option for database in OConsoleDatabaseAppTest
 Update TwitterRestClient to use post method for API requests
 Fixed issue with plugin path and Gears permissions check in WebSettings . java
 Added check for isInEditMode (  )  before measuring the view
 Changed CellChange . java to include clearPrecomputes (  )  method for column model . 
 Added static constants TRUE and FALSE for PactBoolean
 Added method to set paramNames in DefaultUrlMappingsHolder
Added documentation comment for the ActiveRuleQuery class . 
 Fixed EnglishUdLas value to 88 . 72648417258083 for improved performance in NN dependency parser tests . 
 Added streaming support for REST API responses in GetOnRootDocIT test . 
 Added debug perf tracking to DocumentLoader
 Added check to ensure  ' initialExecutionResourcesExhausted '  is called from the correct thread
 Added new enum value  ' a ' 
 Changed the `Luminance1` ,  `Luminance2` ,  `Luminance3` ,  `Luminance4` ,  and `Luminance5` properties from float to void . 
 Added constructor for MillisType class
 Added import for com . intellij . openapi . actionSystem . AnActionEvent
 Added dohasheq method to Util class
 Added fallback package ID for VoiceRecorder intent . 
 Fixed issue with editability of UserDataProperty for Long types . 
 Added support for downloading CSV datasets with missing source key
 Added support for StatefulEJBLocator in LocalEjbReceiver
 Fixed issue with duplicate code in StrictDuplicateCodeCheck class . 
 Added license headers for Apache License 2 . 0
 Fixed blank ID check in WebUtils . java for improved functionality . 
 Fixed type mismatch in ZWaveConverterHandler . java ,  updated import statements to match the latest protocol version . 
 Fixed issue with closing database connection in MysqlPersistenceService . 
 Fixed bug in color parsing and added exception handling for improved reliability . 
 Added timeout setting for ManagedShutdownStrategyTest
Added testQuantifiedPredicates (  )  method to PredicateTests
 Added removed column ID and value to LayoutTypePortletImpl . 
 Committed changes to VcsLogGraphTable to improve jump to commit functionality . 
 Added support for ITemplateDataElementType in XmlASTFactory
 Changed the type of the `races` field from a `Set` to a `Collection` in `SQLSelectTest . java` . 
 Added support for MultiUserChat rooms  ( XEP - 45 ) 
 Added chronology validation for Instant creation
 Added new method to calculate lanes for straight segments
 Added new subtypes to InputMethodInfo
 Added support for ContextStorageOverride class from io . grpc . override package . 
 Added new input method and updated status bar visibility . 
 Added check for non - readonly flag before wrapping nio buffer in ChannelBuffer . 
 Added visibility parsing context to BuildRuleFactoryParams . 
 Fixed issue with date parsing and added warning for missing date . 
 Fixed parallelism configuration in ExecutionConfig . 
 Added getId (  )  method to PythonTestRunConfigurationsConverter to return the ID of the test configuration type
 Fixed bug in cursor adapter implementation
 Added new method  ' setBindings '  and related changes for KeyMap class . 
 Added method `notifyActionStartedForState (  ) ` and `notifyActionFinished (  ) ` to handle state changes in SlidingLayer . 
 Fixed bug in WebContextLifecycleInterceptor to properly handle bundle lifecycle events
 Fixed bug in invalidate (  )  method to correctly update scrollbars when charts are maximized or minimized . 
 Added check for removed actions in ParallelAction . 
Added input parameter to JavaScriptAggregatorFactory constructor
 Changed implementation of `getStringOrNull (  ) ` to improve efficiency by checking `containsKey (  ) ` before calling `getString (  ) ` . 
 Commit message :  Fixes issue with timestamp values in RowSeq class . 
 Changed import for Theme resource . 
 Added support for MANUFACTURER_SPECIFIC command class  ( 0x72 )  . 
 Fixed formatting in ban list message . 
 Fixed cache key for UserImpl to use  ' username2roster '  cache . 
 Fixed focus after mouse event . 
 Added check for fairness in DiffIterableUtil
 Added OnGenericMotionListener to ReactRootView for handling touch events
 Update to use SPVBlockStore in lightweight SPV mode
 Commit message :  Added new reason count map for InliningFailReasonTopListVisitable . 
 Added check for docked stack ' s fullscreen state before resizing non - docked stacks . 
 Added interface for JavaExtensions methods
 Fixed exception message for non - existent block locations in TieredBlockStore . 
 Changed method signature for `getChildFragmentManager (  ) ` to use `replace (  ) ` instead of `add (  ) ` . 
 Fixed regression bug in ProxyHTTPSPDYConnection
 Fixed bug causing null pointer exception when enabling debug mode . 
 Added new method to setMinimumMasterNodes (  )  to acknowledge cluster settings update . 
 Update AbstractLocalTachyonCluster to use UnderFileSystemCluster instead of null . 
 Fixed exception message in Response . java to include response string
Fixed issue with incorrect ClobOpenCount value in DruidDataSource class . 
 Added support for symmetric encryption in SymmetricEncryptionConfig class . 
 Fixed issue with incorrect rendering of array lengths . 
Fixed assumption of Java 1 . 7 in FileAttributesNio2ReadingTest to ensure correct behavior . 
 Added manual connection to MediaScannerConnection for KitKat devices to avoid disconnect issues . 
 Fixed issue with deactivated items in DownloadIndexesThread ,  updated file names comparison . 
 Fixed encoding issue in IOSCharsetDecoder . java by updating the decode method to properly handle byte arrays of varying lengths . 
 Changed method signature for `ResultSetViewer . java` to improve error handling . 
 Fixed bean definition for CacheManager to make it primary or specify a specific one . 
Added CalendarResourceLocalServiceUtil . addCalendarResource (  )  method to test integration test for CalendarResourceService . 
 Fixed indentation in PyEditingTest . java
 Added support for rejecting stray INVITEs for dead dialogs . 
 Fixed indentation in DashRule . java
 Fixed bug in TrackerService . java by changing return type of doDebugLogging (  )  from boolean to boolean . 
 Added support for when clauses in ExpressionAnalyzer . 
 Added support for bootstrapping resources in BootstrapClassLoaderUtil . 
 Changed `PyBlockEvaluatorTest . class` to `PyBlockEvaluatorTest . class ,  PyDocstringTest . class` . 
 Fixed color wheel radius to improve performance
 Added new line for expiration tracking in Cache class
 Changed handling of historic messages in MultiUserChatTest to only retrieve the first message . 
 Added logic to calculate the target size of the split based on the percentage of the total size and the seed number of the entry . 
KeyguardUpdateMonitor :  updated to handle guest users . 
 Fixed issue with XML parsing in IvyXmlModuleDescriptorParser
 Fixed issue with showing footer in ReaderPostDetailFragment
 Added support for validating certificates in Jetty - Util
 Added support for SDK_INT in ZXingTestActivity . 
 Added nullsLast (  )  method to OrderBy clause in SelectBase test
 Fixed type for Mapper . Context and Mapper . class in LoaderScriptWrapper
 Added method to set player list name
 Fixed unused import warnings in SpringExpressionsParser . 
 Fixed duplicate string literal inspection class . 
 Changed the implementation of the `DataTableTopComponent` to use a new `DataTablesModel` instance for the node and edge tables ,  and updated the `AttributeModel` reference to use the new `DataTablesModel` . 
 Added setting JavaScript enabled for notifications webview activity
 Added method to create StringList from iterable object
 Fixed shader source printing in TestShaderNodes . 
 Added request parameters for min_path_precision ,  algo ,  locale ,  vehicle ,  and elevation  (  # 890dc8d5 )  . 
 Added new endpoint for ActorSDK in Application . java
 Fixed issue where graph was not being saved to disk as requested . 
 Fixed bug where calendar was not correctly resetting to today ' s date when using a specific date . 
 Added immutable set returned by getSeeds (  )  for better performance
 Added private constructor to prevent instance creation
 Fixed compatibility issue with Android 11 by removing unused import
 Added new socket properties to JIoEndpoint class . 
 Fixed issue with double / single quotes in method  ' cleanupStr (  )  '  . 
 Added todo to handle character encoding for saved parsers
 Fixed bug in OperatorMergeDelayErrorTest to correctly handle onNext (  )  calls after an error . 
 Fixed exception handling in InternalStreamConnection to properly handle MongoCommandFailureException . 
 Added import for com . intellij . openapi . vfs . ex . dummy . DummyFileSystem in LightIdeaTestCase . java
Added a new test method ,  faultTest (  )  ,  which is marked as  @ Ignore . 
 Added method to quietly close multiple closeables
 Fixed issue with incorrect hostname in header max - execution - time limitation
 Added configuration for port 5060 in RatpackLauncher . java
 Merged 5 shards in InternalEngineMergeIT test . 
 Added test for not found return of empty timeline . 
 Fixed issue where filter would return false for packets with null from address . 
 Fixed order of values in INSERT  .  .  RETURNING and UPDATE  .  .  RETURNING clauses in InsertUpdateTests . java
 Fixed issue with type cast handling in JavaSmartCompletionContributor . 
 Added setInputType (  )  method to TextEnterer class
 Added import statement for Collections class and updated OrientElementIterable implementation to return an empty iterator when iterable is null . 
 Fixed issue with nonexistent parent call IDs in ParcelableCall class . 
 Changed the `import sun . security . mscapi . KeyStore ; ` line to `import com . navercorp . pinpoint . profiler . context . storage . Storage ; ` to fix a compile error . 
 Fixed bug in ThreadMXBean implementation
 Added method length (  )  to calculate the length of a partial token list . 
 Fixed issue with incorrect import for com . liferay . portlet . mail . ContentException . 
 Fixed issue with primary shard allocation and added failed snapshot status for shard  # 
 Added column and categorical mapping functionality . 
 Fixed issue with resolving required placeholders in ConfigurationClassParser . 
 Fixed issue with method signature in DB . java
 Added explicit specification for remote region in getApplicationDeltas (  )  method . 
 Added CountDownTimer import for improved app performance
 Fixed type safety issue in OQueryOperatorContains . java by casting collection element to Object before checking condition . 
 Improved class loading behavior in OverridingClassLoader with excluded packages
 Fixed exception handling in GroovyDslScript to prevent infinite recursion
 Added new method  ' setupActionBar (  )  '  to  ' AddActivity '  for improved UI functionality . 
 Fixed method signature to match YARN API changes . 
 Added nullability checks for where ,  having ,  and limit clauses in Query . java
 Added weak interning for PackageIdentifier instances
 Added check for vanilla Minecraft kicking the connection
 Fixed FIXME comments in GridSchemaV99 . java and Schema . java
 Added flag for allowed network types :  NETWORK_BLUETOOTH  ( 1  <  <  2 ) 
 Fixed license year in FreeMarkerConfigurationFactory . java
 Added clearCapabilities method to builder and updated corresponding documentation . 
 Added new line of defense against failed MetaStore connections by randomly swapping the first element of the metastoreUris array . 
 Refactored VirtualMachineMetrics class to improve readability and maintainability . 
 Reloaded resources due to addition of a WAR file . 
 Added mText field to TextView class for improved performance
 Fixed exception handling for subjectAltName decoding in TLSStreamHandler . 
 Added toString (  )  method to Node class
 Fixed exception handling in Utils . java
 Fixed bean creation arguments support in SimpleJndiBeanFactory
 Added mIWallpaperEngine . mReqWidth and mIWallpaperEngine . mReqHeight fields to handle desired size changes
 Fixed bug in RuleEngine . stateChanged (  )  to properly handle state updates for simple items . 
 Added context specific Weld container reference for Jetty deployment . 
 Added TO DO remove Debug listener in case of disconnect by QUIT or ERROR
 Changed how HTML separators are handled to improve performance . 
 Added new method name check for SuspendableHelper
 Fixed issue where some tokens were silently skipped in the unification process . 
 Added edge label validation for NaturalLogicAnnotator . 
 Added method for creating a template file for raw log buffers . 
 Added config . getTouchEventsForLiveWallpaper check for SDK 7 or higher . 
 Added  ' Change Details '  action to ChangesViewManager
 Fixed bug in sequence tracking for noProgressTracker mode . 
 Added dependency on rstudioapi package  ( version 0 . 4 ) 
Added import statement for MySQL driver in KillbillServerModule  (  # 1 ) 
 Fixed issue where unshare operation would fail if file was deleted before . 
 Integrated Salesforce endpoint test with updated login URL and API version . 
 Fixed handling of line endings in FileMetadata class
 Added support for displaying TV contents in TvView . 
 Added wait for ES to be ready before starting Sonar server
 Added scheduled ping check for NettyTransport
 Added check for closing panel when not expanding due to false peek . 
 Fixed bug in Grails Bootstrap to properly initialize session holder . 
 Fixed proxy settings for GithubConnection class . 
 Added new dependency to ExternalExtensionsDialog class . 
 Added getUfsPath (  )  method in TachyonFile class
 Added singleton annotation to Navigator class
 Added validation summary to EditTextPreferenceWithValidation . 
 Changed import statements in HasOtherAnnotations . java to align with latest AndroidAnnotations version . 
 Added CachingPatternSpecFactory to prevent scanning for class names in String literals
 Added deprecated  @ deprecated tag to field  ' connector '  in Response . java
 Added label for password in FirstWizardPage
 Changed the `checksSpelling (  ) ` method to include a new parameter to suppress misspelled suggestions . 
 Added logging for when a restriction is checked instead of a hook
 Changed behavior of SearchTextField . setSearchIcon (  )  to use native Leopard search control when available . 
 Changed how backpressure buffer capacity is updated to match the new API . 
 Added new method to handle drag and drop operations on source nodes . 
 Fixed bug in View . dumpCapturedView (  )  to correctly handle null views . 
 Added method  ' removeTestProvider '  to LocationManagerService ,  removing MockProvider instances and reinstating real providers . 
 Commit message :  Fixes issue with incorrect average frame consuming time calculation and updates rendering configs to improve performance . 
 Fixed bug in handling cluster topology updates to prevent infinite loop of registration and unregistration . 
 Added configuration for JavaDocLocalInspection to enable highlighting of method parameters and return types . 
 Added toString (  )  method for SystemLogRecord class
 Added logic to save / remove cookies based on expiration date . 
 Added output symbol handling for TableScan nodes . 
 Added didRequestLayout (  )  method for improved performance . 
 Changed how symbol resolution is handled for qualified name references . 
 Added support for decoding Frame objects
 Fixed pattern matching for JSP files and updated language key pattern . 
 Fixed bug in StreamingScalabilityAndLatency test by adding missing `StreamingMode . STREAMING` parameter in LocalFlinkMiniCluster constructor . 
 Fixed bug in TileLaser initialize method to correctly initialize laser data  (  # 518 )  . 
 Fixed bug in inMessage handling
 Added static method to setKeyStorePassword and updated instance variable for OtrAndroidKeyManagerImpl . 
 Added check for no data set in BarLineChartBase . 
 Fixed NullPointerException in ShadowActivity . java by adding Thread . dumpStack (  )  to handle missing setContentView (  )  call
 Added remote video provider option to setVideoProvider method
 Added socket timeout parameter to MongoOptions class
 Added task ID   +  id  +   to the list of tasks to start . 
 Added new method doFlushByteBuffer ( ByteBuffer )  to handle flushing of buffers
 Fixed exception handling for non - supported file formats in IndexUploader . 
 Fixed NullPointerException when setting image by updating super . setImageDrawable (  )  and added error handling for when the image cannot be loaded . 
Fixed issue with displaying bookmark selection toast correctly  (  # 86d0409d1 )  . 
 Fixed issue with parsing properties ,  now throwing a RuntimeException instead of exiting with code 1 . 
 Changed behavior of Animator objects in AnimatedVectorDrawable to improve performance . 
 Changed from SOFT_CONCURRENT to CONCURRENT in PageCacheRecycler . 
 Fixed issue with Postgres refcursor types in AbstractRoutine . java
 Added support for  ' a '   ( Away )  user mode and improved presence monitoring using ISON ,  WATCH ,  and MONITOR . 
 Fixed handling of third - party reconnect / reassociate commands in WifiStateMachine . 
 Update toJSON method to exclude fetchPlan :  *  :  - 1
Fixed issue with first and last leg start times not matching itinerary start and end times . 
 Changed behavior of readHistoricalDataImpl (  )  to read data from a file off the UI thread instead of within the UI thread . 
Added texture binding for PShape3D in both Android and Java implementations . 
 Fixed issue with large image sizes in PixmapPacker . 
 Added unaryCall method to InteropTester class for testing grpc requests
 Improved trace logging in StandardIDPool for better debug output . 
 Changed implementation of `DefaultIdeaContentRoot` to include `Serializable` interface . 
 Fixed maximum memory for biggest query to 8 MB
 Fixed issue with FindResult model implementation . 
 OverlayConfig updated with changes from b893953
 Added check to ensure current page is valid in TitlePageIndicator
 Added method  ' onReset (  )  '  to stop battery receiver and set it to null
Fixed constructor injection for DatabaseEditorInputFactory . 
 Added dismissWithAnimation (  )  method for more control over animation . 
 Changed method signature to remove unnecessary parameter . 
Added import for org . jetbrains . annotations . NotNull in PsiTestUtil . java
 Fixed null pointer exceptions in AbstractJPATest test methods
 Added method to return current recipe for TileAutoWorkbench
 Added method to mark last write timestamp and set active state to true in IndexShard . 
 Added support for remote tunneling with host : remotePort and host : localPort servers . 
 Added method `sendToTarget (  ) ` to `ShadowMessage` class
 Added User - Agent header for ApacheAsyncHttpProvider
Fixed fatal protocol violation and transport errors in HttpUtil . java . 
 Added semaphore with initial count of 0 to RedissonLockEntry class
 Changed implementation of `DeprecatedLzoBinaryBlockRecordWriter` to use `BinaryBlockWriter` instead of `writer_` . 
 Fixed issue with relative time range parsing
 Added check for empty or null hostname in authenticateDomain (  )  method . 
 Added configuration for XZ codec for working with  *  . xz input files
 Fixed bug in PacketTransceiver implementation
 Refactoring AbstractTFS . open ( Path )  to support new internal API  ( TACHYON - 46 )  . 
 Commit message :  Fixes issue where PhoneWindowManager was not properly handling wake keys when screen was off and key was not injected . 
 Added option  - frameworkpaths to allow specifying search paths for custom frameworks . 
 Commit message :  Added new endpoint to test idempotent feature in FtpConsumerIdempotentKeyChangedIssueTest . 
 Added Kryo serialization registrations for improved object serialization . 
 Added new method to reset record before setting identity and from stream
 Fixed formatting issue in StreamError . java
 Added method addComponentItem to handle adding disco items as valid entities for info and item disco requests . 
 Added method getAuthenticationDetailsSource (  )  to AbstractProcessingFilter for SEC - 310 compliance
 Added OpenGL imports to AndroidPreprocessor class
 Added CassandraStorage . java with Cassandra - specific UDFCONTEXT_SCHEMA_KEY and updated logger configuration . 
 Changed how optimistic locking exceptions are handled in `DuplicateVariableInsertTest` . 
 Updating OClass . java to use updated schema version . 
 Fixed issue with tokenization of  ' spiegelt '  and  ' spiegeln '  in the  ' WiederVsWiderRule '  class . 
 Added method getTargetToken (  )  to StorageLoadBalancer
 Added new method to handle null input in GzipUtils
 Implemented non - guava ImmutableList in ImmutableList . java
 Fixed version code auto - increment in ManifestUpdateMojo . 
 Fixed encoding of successData in SASLAuthentication class . 
 Fixed issue with navigation offset correctly overriden in TargetElementUtil . 
 Added getter and setter methods for modCount to ORecordLazyList . 
 Added support for real - time segments in FilterPlanNode for improved performance . 
 Changed MediaRouteDescriptor . java to include a new method to get whether the route can be disconnected without stopping playback . 
 Added cloneForSpatial method to PhysicsHoverControl class
 Fixed hover start value in CoreProxy . java
 Fixed issue with NO_UIPRINT environment variable in InstallDirect . java
 Added constructor arg value at index 169  ( deprecated since Spring 2 . 5 in favor of addConstructorArgValue )  . 
 Added new entry point size of 100 and updated entry point threshold factor to 1 . 5f in OTreeMapPersistent . 
 Changed the implementation of `toString (  ) ` to use the same format as the rest of the class . 
 Changed method `onFailure` to return a `StringRestResponse` instead of an `XContentThrowableRestResponse` . 
 Added nullability check for abstract classes . 
Added support for rate options in GraphHandler . java
 Changed how the `EduStepicConnector` class handles the `requestBody` field to include the `lessonId` parameter . 
 Added method chain . doFilterEvent ( request . getEvent (  )  )  for comet handling in StandardWrapperValve . java
 Added empty wallet creation method with randomly chosen seed . 
 Commit message :  Fixes bug where triangulation produces incorrect results for non - simple polygons . 
 Added support for managing expiring connections as their timeouts occur  ( if configured )  . 
 Added import for org . codehaus . plexus . interpolation . os . Os in TestMetrics . java
 Added ignoreUnmapped parameter to FieldSortBuilder
 Changed behavior of `contains (  ) ` method to correctly handle null inputs . 
 Fixed issue with invalid FQ names . 
 Reloaded local message list for folder
 Fixed bug where ShellException was not properly handling inner exceptions ,  now properly subclassing and including stack trace . 
 Added support for YUV_420_888 format in Type . 
 Added generateHeaderId method in UltimateViewAdapter
Added support for Enterprise mode in `EnterpriseNeoServer . java` . 
 Added empty iterator for EmptyColumns . 
 Added check for null cause in DefaultPromise # setFailure0
 Fixed formatting for date parsing . 
 Fixed issue where language tool was not properly shutting down in LanguageToolSupport . java
 Added getCullingArea (  )  method and updated corresponding documentation in Group . java
 Changed how `withWorkDirectory` method handles Unix environments to use `FileUtil . toSystemDependentName` to get the absolute path of the work directory . 
 Added pause (  )  method to ShadowFragment class
 Added Boolean anonymous field to AuthToken class
 Fixed bug causing InterruptedException in SpiceManager . 
 Added support for the ancora option . 
 Fixed bug in RabbitMQ consumer implementation . 
 Fixed bug in toArray (  )  method to properly close the cursor when batchSize is exceeded
 Changed behavior of `AddLocalExchanges` optimization to pass through parent preferences for filtering sources . 
 Added support for converting raw bytes to hex strings . 
 Added legacy support for restored pipes
 Added configManagerConfig to MasterMain for JSON serialization
 Generated new BitmapFontData for size 58 , 6 pixels . 
 Fixed exception message formatting in Driver . java and Utilities . java
 Added new line of code to handle pause / play media player
 Added final instance to NoOpProjector class
 Added todo comment for new Windows agents . 
 Commit message :  Fixes error handling in HgErrorHandler to properly handle null results . 
 Added init method for aggregations in PercolatorService . 
 Added flush (  )  method to ResponseBodyEmitterReturnValueHandler for improved performance
 Addedprintln statements to MotionTrack for debugging purposes . 
 Added null check for callback in onPostExecute (  )  to prevent NPE . 
 Changed method `changeTimeZone` to handle multiple time zones . 
 Fixed bug in ticket handling for LongPoll and Stream requests
 Added trim (  )  method to entryPoints parameter in CsrfPreventionFilter class
 Added localhost (  )  method to PeerAddress class
 Added double precision for current data in SeekableViewsForTest
 Added test for determining the value of an enum using a method
 Fixed issue with folder detection in TachyonFS . java
 Added  ' PyCharmCore '  prefix for Python plugins
 Added logging of IOException with message and cause for ReplicaSetStatus . 
 Added trigger for expanded state of QSPanel . 
 Commit message :  Removed item from roster upon removal request . 
 Added import org . junit . Ignore for ignoring validation of documentation
 Added new policy for active mobile network
 Fixed exception handling for disconnection noise in NetworkDispatcher . 
 Added method convertFieldName (  )  to convert property names in XML files
 Fixed constant values for INTEGRATION_WORKER_RESOURCE_MEM and MASTER_JOURNAL_FOLDER . 
 Changed test to expect config exception for unsupported return type in `TestRestLiResourceModels . java` . 
 Fixed issue with setting HttpOnly and Secure flags for cookies . 
 Commit message :  Added a new border to the VcsCommitInfoBalloon component . 
 Added expected exception check for Test . java
 Fixed unused return statement issue in UnusedReturnValue . java
 Added new metrics for concurrent execution count and health counts in HystrixCommandMetrics . 
 Optimized TextureAtlas . java :  added logging for warning about too small atlas size
 Added support for multiple LocaleList instances in the same application . 
 Changed the implementation of setCursor (  )  to use a new method signature ,  allowing for a more concise and readable codebase . 
 Added method  ' notifyAccessibilityStateChanged '  to AdapterView for ICS and above
 Added check for non - editable text to prevent showing IME when selection is in progress . 
 Added log statements to track spent transaction outputs
 Fixed issue with incorrect file comparisons in GeneratedArchiveBuildOutcomeComparator . 
 Fixed issue with empty baseNames array in PluginAwareResourceBundleMessageSource class . 
 Fixed bug in `jobDone` method
 Added isScreenBrightnessBoosted (  )  method to BridgePowerManager
 Fixed size of MvcEndpoint beans from 6 to 5 . 
 Changed EntityDamagedByEntityEvent to EntityDamagedEvent . 
 Fixed bug in tessellation shader handling
 Fixes issue with incorrect sprite color in FastImage actor
 Added shareWithUser field to OCFile class
 Added support for replication operation type in TransportShardReplicationOperationAction . 
 Added support for serializing edge properties with null values . 
 Changed query string from spaghetti to spaghetti with updated field values . 
 Fixed null pointer exception when feed content is empty . 
 Added import for ExternalSystemException and updated classpath for RemoteExternalSystemCommunicationManager
 Added static logger for TicketGrantingTicketImpl
 Fixed charset nullability in file / fileResource methods
 Fixed compile error in AuthConfigFactory . 
 Changed the color of the disabled state from white to darker gray . 
 Fixed bug causing test to fail with  ' java . lang . IllegalStateException '  . 
 Added skip logic for resolving supertype references . 
 Changed how `isDirty (  ) ` is called to use the `editor` instead of `commandContext` . 
 ResourceShrinker :  updated parsing logic for resourcetxt files
 Added new boolean fields for HA enabled and metrics enabled ,  and updated default values for both . 
 Fixed issue with isToString (  )  method in InvocationTest
 Fixed exception handling in HeapAnalyzer to properly capture and report exceptions . 
Fixed issue with property block size calculation in WriteTransaction . java ,  improving performance . 
 Added test root directory to files to delete in GitPlatformTest . 
 Fixed issue where minimum idle connections were not properly enforced due to negative or invalid values . 
 Added import for android . annotation . SystemApi and updated setDisableWallpaperTouchEvents method to use it . 
 Changed how the number of compute nodes is reported in the HTML documentation . 
 Fixed bug in WifiStateMachine . java by correcting null pointer exception in disableInterface (  )  method . 
 Fixed maximum deviation percentage for secondary reads from 3 %  to 5 %  . 
 Added PlaceBuffer observation for getPlaceById (  )  to fetch places from the Places API . 
 Changed how GeneralVisualCppCompileSpecToArguments . java handles position - independent code . 
 Fixed bug in PGL . java by updating backTex variable in else if statement . 
 Added getUserId (  )  ,  getAppId (  )  ,  and getRunId (  )  methods to WorkflowId class
 Rewrote ClosureRewriteClass to use hotSwapScript (  )  instead of process (  )  for improved performance . 
 Added method  ' getSuccessDescription (  )  '  to PyPackageManagerUI
 Changed how `myTable . setValueAt (  ) ` is called to fix Mac - specific issue with `getCellEditorValue (  ) ` being called before `myValue` is set . 
 Added error for  @ export in non - legacy Google modules
 Fixed buggy thread count calculation in CacheUpdateRunner
 Added support for multiple genres in TvContract . java
 Added method to retrieve resource entry name for views with invalid IDs . 
 Added support for BrowserCompatHostnameVerifier in Ion
 Added saveInstanceState method to ControllerActivity
 Fixed issue with residual deviance calculation in GLMBasicTestBinomial . java
 Added fluid rendering to GuiAdvancedInterface
 Fixed incorrect mod list display in GuiMissingMods . java
 Added check for themes directory in PluginsEnvironmentBuilder
 Fixed issue with file path resolution in GrailsResourceUtils class . 
 Fixed formatting issue in ReplicatedTransaction . java
 Fixed gesture detection for WebView to handle two fingers close together for scrolling
 Added getClientId (  )  and getColor (  )  methods to ProjectUser class
 Added  ' Comparing '  import for improved code quality
 Added method for getting current system time with user - set offsets applied
 Fixed sleep time to avoid CPU throttling in MapStoreTest . 
 Added hideKeyboard (  )  method to MessageViewFragment for improved user experience . 
 Fixed bug in activation3Reverse calculation
 Added edge label for unbound dependencies in GraphVisualizer
 Added Linux - specific Java SDK path
 Added method to get current tile index in TiledSprite class
 Changed from using a cached thread pool to a single thread pool . 
 Added support for ElevatorEdges in StreetUtils
 Added PsiFile implementation for HeavyIdeaTestFixtureImpl . 
 Changed the visibility logic for activities launching behind other activities . 
 Fixed bug in Intent . java related to activity stack clearing when task is reset . 
 Added note about including library - specific code in agent JAR file for better advice location
 Update VERSION to 1 . 3 . 1
 Added ctx . alloc (  )  to allocate composite buffer for message aggregation . 
 Changed implementation of `LanguageConsoleBuilder` class to improve performance . 
 Fixed issue with incorrect number of shards in test case . 
 Added test to verify persist functionality in JpaUsePersistTest
 Changed implementation of `setEnabledStateOnViews` to use `holder . itemView . setFocusable (  ) ` instead of `holder . setDividerAllowedAbove (  ) ` and `holder . setDividerAllowedBelow (  ) ` . 
Added Apache License v2 . 0 to NettyRedirectConnectionUsageTest . java
 Fixed license year in FileBasedDeclarationProviderFactory . java
 Added method getPersistedConfig (  )  to persist time range configuration . 
 Added nullable return type to isUnsaved method
 Added hash of browser cookies to Request object
 Added method to retrieve phone number from shared preferences
 Added method `orElseThrow ( Supplier ) ` to Value . 
 Added support for AppCompatActivity in ListSamples . java
 Fixed bug in KeyParser . java by updating shortcuts array size to match actual number of shortcuts . 
 Changed how `DataFormat` is compared in `testGetDataFormatFromCxfEndpontProperties (  ) ` to match the new value added in `b / components / camel - cxf / src / test / java / org / apache / camel / component / cxf / util / CxfEndpointUtilsWithSpringTest . java`
 Added  @ Deprecated annotation to  ' resources '  and  ' getNamingResources (  )  '  method in ContextTransaction . java
 Fixed bug in WebURIStat . java by updating  @ Profiler annotation . 
 Fixed issue with empty declared name ,  now using fieldInfo . declaringClass (  )  . name (  )  . toString (  )   +   /   +  fieldName for localContextName

 Fixed bug in Surface initialization
 Fixed scroll changed event handling for improved performance
 Added override implementation for equals method in GetStatusOptions class
 Changed implementation of `RegexpHeaderCheckTest` to remove unnecessary empty constructor and added a new test case for `setHeader` method . 
 Added version string to HeaderSymlinkTreeHashCode calculation
 Changed logging levels for certain messages . 
 Added method to open raw resource files with custom file name prefix
 Added long hash function for buildKeyName (  )   (  # 57dd141 ) 
 Added padding API for CompoundButton in Jelly Bean MR1 and above . 
 Changed ApplicationConfig interface to include new init parameter for using servlet context parameters . 
 Changed test to use `BlockId . createBlockId ( 0 ,  BlockId . getMaxSequenceNumber (  )  ) ` instead of `MAX_SEQUENCE_NUMBER` in `createBlockIdWithMaxSequenceNumberTest (  ) ` . 
 Fixed issue with distance calculation in VoiceRouter class
 Added import for ResolveResult class and updated getTargetElement method to handle ResolveResult objects . 
 Fixed issue with incorrect node join completion count in testJoinWithManyNodes ( true )  . 
 Added equality check for foreign key columns
 Fixed license year in ActionButton . java
 Added support for Jakarta Regexp capabilities in RegexQuery
 Added setHighest (  )  method to GCResistantCache class
 Fixes bug in RecordLoad . NORMAL acquisition for ReadNodeRecordsByCacheStep
 Fixed bug in account comparison logic . 
 Fixed namespace prefix in XML element name
 Added memberAlive method to handle member addition
 Added  ' true '  to the  ' setId '  method to ensure delta tracking for sessions
 Added method chaining support for FileChannel . position and FileChannel . truncate
 Added support for executing task operations with JSON payload
 Changed how actions are preloaded to improve startup performance . 
 Changed implementation of `visitChildrenRecursively` from a lambda to a method reference to fix a compile error . 
 Added support for non - local files in DiffContentFactoryImpl
 Added support for translating classes . dex entry . 
 Fixed cancel (  )  method to return true on cancellation
 Fixed issue with PDFView setInitialView function in PdfJsWindow . java
 Added method  ' setSelectedTrack ( int ,  int )  '  and updated caption listener handling for TYPE_TEXT tracks . 
 Added new `Mult` import in MultiMap . java
 Added import for com . ning . billing . usage . glue . UsageModule in TestJaxrsBase . java
 Added deep measurement for size calculation in ConcurrentLinkedHashCache
 TypingNotificationEvent class updates for improved toString (  )  method . 
 Changed how refs are handled when they are not found in local or remote branches . 
 Added exception for forcing ephemeral users on systems without split system user . 
 Changed the `HashMap` to `ConcurrentHashMap` in `statistics` to improve performance . 
 Fixed issue with number of hidden layers in NeuralNetConfiguration . 
 Resolved a potential infinite recursion in `isInheritor (  ) ` method . 
 Fixed issue with contact group name in OperationSetPersistentPresenceIrcImpl . 
 Changed how mime types are registered for png files in StaticFilesTest . java :  added new registration for image / png and removed old registration for custom - png - value . 
 Changed how assertTrueMessage is created to include the message parameter . 
 Changed tooltips for buttons in QuickMenu to display contactInfo instead of userInfo . 
Added lengthSquared field to DenseVector for better performance
 Added  @ Deprecated annotation to TypeVisitor class . 
 Changed scroll pane policy to improve performance . 
 Added check for empty circuit list before throwing PeerDiscoveryException . 
 Changed Platform . getNativeLibraryResourcePrefix (  )  to use bsd - x86 for OpenBSD ,  FreeBSD ,  and GNU / kFreeBSD ,  and android - arm for Android . 
 Added check for visible activities in ActivityStack . java
 Fixed bug in testCollapserUnderConcurrency (  )  to handle exception correctly
 Fixed issue with back key behavior when sheet is showing . 
 Commit message :  Modified MvcFramework . java to add a new method for creating default directory structure in a module . 
 Fixed bug in ATNConfigSet . toArray (  )  method by returning configLookup . toArray (  ) 
 Fixed issue with email extra in DebugLogs activity . 
 Fixed exception handling in OBinaryNetworkProtocolAbstract to correctly log unmarshalling errors . 
 Added wrapper class for variable expressions
 Added import for org . elasticsearch . index . analysis . AnalysisModule in AbstractQueryTestCase . java
 Added logging statement to skip delivery of hint for deleted columnfamilies . 
 Added support for concurrent map operations in ConcurrentIdentityHashMap . 
 Added default access decision voters for InterceptUrlConfigurer . 
 Initialized Go module libraries after project opening
 Fixed issue with lightweight edges
 Fixed NullPointerException and IOException handling in HttpResponse class . 
 Added field initialization in HighlightBuilder class
 Fixed bug in Java class loading for improved performance
 Added new rectangle for content rect in WindowManagerService . 
 Fixed issue with empty result mappings for nested result sets . 
 Added support for JpsAndroidModuleExtension in AndroidProjectBuildTarget
 Changed handling of websocket classes to allow webapps to use classes from system classloader . 
 Added property MAX_STACK_SIZE with default value 2 . 
 Fixed nested connection class to remove unused import
 Fixed issue with gram matrices exceeding memory limit in GLM model training . 
 Added attribute  ' code '  to  ' status '  element in LocalMUCRoom . java to support MUC status codes  ( 100 ,  201 ,  210 )  . 
 Added deprecated method  ' setModel ( ListTableModel < Item >  )  '  and updated documentation . 
 Fixed issue with handling key events correctly . 
 Added return type check for Identity class
 Added toString (  )  method to return the query string for encoded URIs
 Added method  ' initWorld ( String title )  '  to generate a new random seed for the world if it ' s not present on disk . 
Added support for WebSocket transport in AtmosphereResourceImpl ,  and updated FrameworkConfig to use the correct transport based on the request . 
 Added  ' onStart (  )  '  method to track view entry
 Changed implementation of `FilePartSource` to throw a more informative exception when the file is not a normal file or not readable . 
 Added context import in StandardContextValve . java
 Changed how missing values are inserted in frames . 
Added bind exception handling in NettyUtil
 Added new method to handle  ' location '  equiv in HTML content
 Fixed issue with repeated requests being queued for deletion . 
 Added import statement for MediaPlayer class
 Added new language key set method in Languages class . 
 Fixed boolean return type in hasShareContactsEnabled (  )  to match the preferred value
 Fixed binary search algorithm for token identification . 
 Update default subscription processing mode to SUBSCRIPTION_ACCEPT_ALL
 Changed the code to fix a potential SocketTimeoutException issue . 
 Added import statement for com . facebook . buck . event . ConsoleEvent in GenAidl . java
 Fixed version calculation in TodoIndex to improve indexing performance . 
 Added ClassClassPath to Weaver constructor for serialization
 Added import specs for GoFile . 
 Added check for transit mode in qualified mode set sequence for car modes . 
 Added new metadata binding for  ' function - signature '  ,   ' name '  ,  and  ' name - given '  in ReferenceProvidersRegistry  ( commit 543495d )  . 
 Fixed issue with offline total document count calculation in ValidationManager . 
 Changed how ReplyException is created to include the message parameter in the constructor . 
 Added check for exceptions in XField class  ( 4600336a -  > 3f4c70cc )  . 
 Formatted visitor and view counts in StatsDetailsActivity using FormatUtils . 
 Fixed issue with alpha value in PhoneStatusBar . java ,  ensuring proper animation and avoiding uninitialized memory . 
 Added method to wrap Promise in SPDYClient
 Changed the `fileRoot` import statement to use the `file` method instead . 
 Added constructor for FolderObserver class
 Added support for saving DBW handler configurations for connections in the ConnectionPageNetwork . 
 Added worker hook check for null argument in TopologyBuilder . 
 Added deprecated methods for WrapType constants . 
 Fixed compatibility issue with BaseAdapter . 
 Added logger for InputFileBuilder class
 Added method `previous (  ) ` to retrieve the previous item in the list . 
 Added synchronized method for adding elements to the immutable set in DefaultPicoContainer . 
 Added new options for Scala compilation . 
 Changed the lock level of the `writableScope` field from `WritableScope . LockLevel . READING` to `WritableScope . LockLevel . BOTH` . 
 Changed handling of exceptions to include the message . 
 Added new method to add animated gesture to views ,  including offset start and end positions . 
Added import for PackratRestoreActions in Packrat . java
 Changed Player . java to add getExperience (  )  and setExperience (  )  methods ,  as well as updated getTotalExperience (  )  and setTotalExperience (  )  methods . 
 Update plot publish MRU list
 Added support for comparing namespaces with supertypes
 Added optimization for PsiModificationTracker in PsiCachedValue . 
 Added  ' presenter '  field to Participant entity class . 
 Commit message :  Fixes issue with incorrect statistics logging . 
 Fixes issue with drop shadow not being drawn in some cases
 Changed `MapUtil . map (  language ,  builtin ,  name ,  none  )  , ` to `MapUtil . map (  language ,  javascript ,  body ,  position . length (  )   >  10  )  , ` in `TraverserFunctionalTest . java` . 
 Fixed bug in Java NIO file watching implementation
 Fixed pattern to detect lines and columns in ChooseByNamePopup class . 
 Added break statement in `MSG_OPEN_RECENTS_PANEL` case for proper visibility handling of recents panel . 
Added new copyright and license information for SonarQube . 
 Fixed unsupported operation exception message in SessionImpl . 
 Added support for listing routes in the  ' RouteListCommand '  class . 
 Fixed message for ColumnNotFoundException in ColumnNotFoundException . 
 Added implementation of containsOrdinal method . 
 Fixed issue with file input stream in SSHUtil . java
 Added support for handling welcome files with paths that start with a slash . 
 Fixes isLayoutRtl (  )  method to return correct result for indeterminate drawable . 
 Added console focus check and send to console functionality
 AdjustArrayRangeAction :  updated range calculation for improved performance  ( 55c53af )  . 
 Added randomized sampling for network statistics histograms . 
Fixed method invocation toString (  )  output to include argument list in Gradle messaging dispatch test
 Fixed issue with incorrect digit handling in MorfologikSpeller class . 
 Fixed issue with restricted extra parameters in XPackageManager . java
 Added empty `isEmpty (  ) ` method and updated size checks . 
 Changed behavior of `JmsDiscoveryTest` to wait for at least 5 seconds before shutting down the Camel context . 
 Added support for suggesting Sdk home path based on sdkType . 
 Added deprecated method  ' performInplaceRename (  )  '  in InplaceVariableIntroducer . 
 Added option to suppress printing of status during execution in AvroExternalJarProgramITCase test . 
 Added check for parameter variables in generated code
 Fixed byte order for element count in DexBackedPackedSwitchPayload . 
 Changed the behavior of the `ExistsRecord` method to return `false` instead of `null` for linked records in `FetchPlanTest` . 
 Added GSM operator IDP string property
Fixed deprecation warning in HashFunction . java and Hasher . java by removing the deprecated method and adding a new method to hash unencoded characters . 
 Added title to VizGui frame for improved user experience
 Changed how SearchEverywhereAction handles empty search patterns . 
 Changed the layout of the OptionsEditor panel to add a margin of 1 pixel on all sides . 
 Rewritten PerIndexDocumentMap class to fix agent authorship issues
 Fixed content type header for INVITE requests to be application / sdp ,  as reported by Oleg Shevchenko  ( Miratech )  . 
 Fixed encoding argument in TextEditingTarget . onResponseReceived to prevent errors . 
 Added new hooks for SIP API support and call notification in Meta . java . 
 Fixed bug causing scrim view background alpha to reset on orientation change  (  # 3355 )  . 
 Fixed bug in UriBeaconDiscoveryService by updating serviceUuid for proper filtering . 
 Fixed null pointer exception in WebView . java when handling selection copy . 
 Added flag attribute to IntDef annotation
 Fixed interrupt status to prevent SkyFunctions from doing anything fancy after error bubbling . 
 Changed how sub - executions are found in `SetProcessDefinitionVersionCmd` to use `findChildExecutionsByProcessInstanceId (  ) ` instead of `findChildExecutionsByParentExecutionId (  ) ` . 
 Added final modifier to response array for improved code readability
 Fixed menu refresh ID for TagViewFragment
 Fixed context usage in DemoApplication to properly initialize FileDownloader . 
 Added support for clearing the background color in RenderPass . 
 Fixed bug in DefaultGradleLauncher . java by updating the error handling logic . 
 Fixes layout position calculation for views . 
 Added check for non - suspicious elements in UnusedDeclarationInspection . 
 Fixed bug where map . set (  )  was not being called with the correct time unit . 
 Fixed issue with Actions not being updated correctly in NotificationMediaTemplateViewWrapper
 Added new method to NotificationService for scheduling notifications
 Added method  ' show (  )  '  and  ' hide (  )  '  to Screen interface for better API usage . 
 Changed the logging message from `warn` to `debug` to indicate a blocking I / O read operation from a GET request . 
 Changed implementation of `ItemStack` class to remove unused `Serializable` interface . 
 Fixed issue with localhost and loopback address detection in Proxy . java
 Fixed payment failure handling in TestBusHandler
 Fixed bug in TestGraphProperties . java by updating File storeDir variable to use getAbsoluteFile (  )  method . 
 Added animation support for PicassoDrawable
 Fixed bug in databaseSegmentManager . stop (  )  method
 Fixed bug in handling next counterpart for encrypted messages . 
 Update to implement Bukkit version checking
 Update WXEnvironment . java to use WXSDK_VERSION =  ' 0 . 6 . 2 . 2 ' 
 Fixes scrolling behavior in Scroller class . 
Fixes issue with ANY expression in LuceneQueryBuilder ,  ensuring that reference symbols are properly handled . 
 Fixed buffer size in ResponseBodySubscriber to improve performance . 
 Fixed UnsupportedOperationException in MessageList . remove (  )  method . 
 Added method to show active security policies in Main . java
 Fixed bug in notify method to properly reset last notification . 
 Added yaw parameter to Location constructor
 Changed `iv . setScaleType ( ImageView . ScaleType . FIT_CENTER )  ; ` to `iv . setScaleType ( ImageView . ScaleType . FIT_CENTER )  ; `
 Changed how plugin is installed and removed in PluginManagerTests . 
 Changed behavior of `IdeEventQueue . isFocusTransferInProgress (  ) ` to return `IdeFocusManager . getInstance ( project )  . isFocusTransferInProgress (  ) ` instead of relying on a local variable . 
 Fixed bug in handling file paths
 Added ability to save output to file in TestObject class
 Added toString (  )  method to ActionWrapperBase for easier debugging
 Changed the `startDate` comparison in `TestTransfer` to compare it to the `beforeMigration` and `afterMigration` dates ,  respectively ,  with a margin of 2 months added to each date ,  to ensure correctness after the migration . 
 Added privilege ALL_PRIVILEGES to CreateShareWithShareeOperation for better user experience
 Fixed dismissNotifications (  )  method to properly handle connection state changes . 
 Added fileMmapCleanerHackEnable (  )  for compatibility with recent MapDB versions
 Fixed formatting issue in BuilderStatisticsMessage constructor
 Update to use  ' content_pack . json '  instead of  ' configuration_bundle . json '  for attachment filename . 
 Added relative path to Jetty root directory in TestServer . java
 Fixed issue with empty values for PostgreStructValueHandler . 
 Fixed issue with NED archive download path . 
 Added PathUtil . findRtJar (  )  to the classpath for improved performance . 
 Fixed issue with parsing exception in AbstractSortTestCase . java
 Added GetGlue object id field to SeriesGuideContract for supporting shows without IMDb id  (  # 128 , 8 ) 
 Fixed issue with idempotent ref test . 
 Added import statements for ConfigFilterTest and HttpConfigLoaderTest1
 Added a new break statement in the `case UNDEFINED` branch of the `AgentBuilder` interface . 
 Fixed exception handling in onPreExecute (  )  method . 
Added import for Charsets in MapperService . java
 Added new commit range handling logic for tags and orphan branches . 
 Added indeterminate progress bar visibility check for ActionBarSupportImpl . 
 Added toggle for disabling missing model text zoom in the client gui . 
 Changed synchronization behavior for `downloads` list in `DownloadService` . 
 Added support for storing plugins in SMB repositories
Rewritten EffectivePredicateExtractor to handle deterministic conjuncts  (  # 6169632 )  . 
 Commit message :  Fixes  # 123456 :  Corrects for rare floating point imprecision in EarClippingTriangulator . java to prevent incorrect area sign calculation . 
 Fixed issue with finding indexingService ,  now checking for correct configuration . 
 Fixed case conversion for SQL completion in SQLCompletionProcessor
 Fixed issue with unread message count display
 Added documentation comment for nested types in JavaGenerator class
 Fixed bug in WebAppContext . java by changing _copyWebInf from true to false . 
 Added method to return current load balancer ' s service properties
 Added post processing calls for slots that don ' t need to be built but still need post processing . 
 Added hide annotation to destroyHardwareResources method
 Fixed test failure on CI servers by ignoring it with  @ Ignore annotation . 
 Added ExitCodeGenerator to BatchAutoConfiguration
 Fixed deprecated method signature in InputMethodUtils . 
 Fixed license year in AnnotationConfigUtils class . 
 Fixed issue with concurrent scanning of files for indexing . 
 Added OnePixelSplitter for improved layout in OptionsEditor
 Added support for PsiFile in AntFileImpl
 Added ability to read input from file instead of stdin in Runner . main (  ) 
 Fixed rendering issue with text antialiasing . 
 Added deleteQuietly method for cacheFile to prevent exceptions when moving file . 
Added read - only flag to TestFileResourceSetVirtual class
 Renamed PyMagicLiteralRenameHandler class to align with new naming convention . 
 Fixed indentation in Translator2AndroidXml . java
 Added check for ActivityManagerNative . isSystemReady (  )  before sending ConfigChangeBroadcast
 Changed the JAXBConvertTest to handle NoTypeConversionAvailableException for stream caching . 
 Added check box for sending usage data in WelcomeDialogFragment
 Added new method  ' testTypedefs '  to test Cython ' s support for typedefs
 Fixed bug in handling of HTTP responses
 Fixed null pointer exception in MultiUserChat . enter (  )  method . 
 Added instance profiling data for JobProfilingData :  2 new instances . 
 Changed file contents to include new `CountsStore` instance for dump file . 
 Added support for custom configuration sources using the `add (  ) ` method . 
 Added setBlockCommentStartDelimiter and setBlockCommentEndDelimiter methods to EmbeddedDatabaseBuilder for improved configuration . 
 Fixed handling of directory URLs with trailing slashes in HttpStaticFileServerHandler . 
 Added ListView . CHOICE_MODE_SINGLE to allow single selection of notifications . 
 Added popup hiding method for improved performance
 Changed the `attributeModel` initialization to use `Lookup . getDefault (  )  . lookup ( AttributeController . class )  . getModel ( graph . getGraphModel (  )  . getWorkspace (  )  )  ; ` instead of `Lookup . getDefault (  )  . lookup ( AttributeController . class )  . getModel (  )  ; ` . 
 Added new wrapper for socket  # 1234567890 ,  updating read / write handling and poller flags . 
 Fixed issue with trip ID and agency ID for non - transit legs . 
 Added import for com . intellij . openapi . progress . ProcessCanceledException . 
 Added ability to group dependencies by scope type
 Changed method `getHelpTopic (  ) ` to return reference . settingsdialog . codestyle . java instead of null . 
 Fixed bug in DataSupport . where (  )  method
 Added runtime classes for test compilation
 Fixed issue with XML parsing in ServerManager class
 Changed plugin manager test to allow extra files . 
 Added link to context menu developer guide in ActionMode . java
 Fixed null pointer exception in Realm . java by updating counter variable . 
 Added new SkyFunctions to the package locator dependent values :  AST_FILE_LOOKUP ,  FILE_STATE ,  FILE ,  DIRECTORY_LISTING_STATE . 
 Fixed bug in method invocation for serialization
 Added  . compile for all exceptions of the element in AbstractPatternRule . 
 Added logic to insert row ID to newly inserted rows
 Added license comment to NettyAsyncResponseTest . java
 Added ReaderUserActions . setCurrentUser ( response )  to the  ' AUTHENTICATE_USER '  step . 
 Added import for Google Guava ' s Lists class
 MaterialConfigsTest :  updated validation message for autoUpdate attribute
 Added support for displaying home as up enabled in action bar
 Added system look and feel class to GuiMode
 Added print statement after test method in MatrixJNITest . 
 Added neutral button for immediate publication in EditPostSettingsFragment
 Changed ServletUtil . java to support internationalization by adding a template for the content disposition header . 
 Changed the relationship between `c` and `d` in the `PathRepresentationTest` class . 
Fixed issue where  [ match_phrase ]  query doesn ' t support multiple fields ,  added validation for multiple fields . 
 Added setWillNotDraw ( false )  to BlurView to improve performance
 Update SystemStoreConstants to remove hardcoded zone replication factors and replace with a single replication factor for all zones . 
Added enablement of deleting locally added files in RollbackChangesDialog based on existence of new files in the change list  (  # 121 )  . 
 Fixed getFileSeparator (  )  method to return  '  /  '  for FTP files . 
 Added headers for CORS support in CorsInterceptor class
 Added method setLastColumnNumber (  )  and getLastColumnNumber (  )  for EditorEx interface  ( deprecated in IDEA 16 )  . 
 Added state to normalItem
 Added new case for CONTROL_AWB_MODE_SHADE and updated white balance values for all modes
 Added optimization for OMVRBTreePersistent :  removed unnecessary nodes based on distance threshold . 
 Added support for connector  ' connectorName '  . 
 Added return statement to canDrain method in BlockFluidClassic
 Changed the `nodePropertyFilter` method to accept a `nodeProperties` string instead of a `String [  ] ` argument . 
 Improved performance of TestPerformance . java by updating listenerClass from BlankJavaListener to JavaBaseListener . 
 Added ability to iterate over empty result sets in ResultSetIteratorTest . 
 Rewritten ClosureRewriteModule with new reportUnrecognizedRequires (  )  method
 Update PyUnboundLocalVariableInspection to enable by default
 Fixed bug in `isCancelled (  ) ` method to prevent infinite loop in `AsyncTaskFragment` . 
 Changed behavior of `ReleasedVersionDistributions` class to handle cases where most recent final version does not contain  - milestone -  suffix . 
 Fixed bug in expand method to correctly expand all items . 
 Fixed native method signature to match Java interface changes . 
 Fixed issue with unpacking WARs and deploying XML configuration . 
 Fixed null pointer exception in CacheStatisticsAutoConfigurationTests . 
 Fixed IllegalArgumentException message in Url . setScheme (  )  for invalid scheme values
 Fixed issue with incorrect status code interpretation in decode method
 StandardVm updated to use Apache License 2 . 0
 Added import for OsmandSettings in VoiceRouter . java
 Added  @ ThreadSafe annotation to ResourcePool class for improved concurrency support . 
 Added new ignored error to Pep8ExternalAnnotator class . 
 Added new implementation of EphemeralFileSystemAbstraction with soft references to buffers for better memory management . 
 Added commit documentation for IpnbFilePanel . 
 Added threshold for sparse set size to improve performance
 Added method to remove fragment from all loader managers
 Added check to ensure decrementing outstandingTasks before starting new splits to prevent starvation
 Added consistency level to AbstractThriftCqlQuery constructor
 Added break statement to prevent infinite loop when data set is null . 
 Added objc_library rule class
 Added BSONObject support for key and value in MongoRecordWriter . 
 Added ShowHiddens action to FileSystemTreeFactory
 Changed how we handle function calls with periods to correctly identify potential method functions . 
 Fixed bug in matrix handling for PShape3D . 
 Added new element  ' affiliation '  to Affiliation class
 Added constructor to accept a version parameter for ZapRelease class
 Added method remove (  )  to RemoteTaskRunnerWorkItem class
 Fixed issue with Perforce command execution and added waitFor (  )  to ensure proper exit status  (  # 9f1931a3 )  . 
 Added support for setting session name in client - side module . 
 Changed handling of requires in ProcessClosurePrimitives to only error if the type has not been provided yet . 
 Fixed IllegalStateException in TaskListActivity . loadTaskListContent ( true ) 
 Fixed line count and compared lines in AbstractPollingReporterTest . 
 Fixed issue with canceling downloads in AddFeedActivity . 
 Added drawable divider for preference list and adjusted divider height in SiteSettingsFragment . 
Added test for missing check in SuppressionsLoaderTest
 Added new action button component to VcsLogQuickSettingsActions . 
 Changed Bek graph sorting behavior to match user preference . 
 Changed how invalid comparator is handled in DatabaseDescriptor . java . 
 Fixed issue with CardView layout in ShadowTransformer class . 
 Fixed issue  # 6425c7 :  Removed unnecessary nocommit call in IndexShard . java
 Changed how the `sharedPrefs . edit (  )  . putInt ( new_quotes ,  0 )  . commit (  )  ; ` line is indented to be in line with the other `sharedPrefs . edit (  )  . putInt (  ) ` lines . 
 Added method signature for setConf (  )  in ForwardingFileSystem to match superclass . 
 Added validation for non - letter and non - digit characters in property names . 
 Commit message :  Fixes issue with audio data processing in SipToFlashAudioStream class . 
 Added new HTTP content compressor for better payload encoding
 Fixed implementation of MyConverters class . 
 Added new readStream field to PumpImpl for improved performance
 Added new serializer for keys and values in HTreeMap . 
 Added non - zk lock manager configuration to prevent HS2 from connecting
 Fixes image placeholder insertion in NotificationsUtils . java by changing Span . SPAN_INCLUSIVE_INCLUSIVE to Span . SPAN_EXCLUSIVE_EXCLUSIVE  (  # 58fc31c11 )  . 
 Merged PredictionContext instances with same graph or equal to each other . 
 Fixed issue with service name mapping in CuratorServiceUtils :  replaced  '  /  '  with  '  :  '  to improve HTTP interface compatibility  (  # 1234 ) 
Added snapshotDisable (  )  method to disable snapshots in DBMaker for performance optimization
 Fixed bug causing list to not display after fragment resume  (  # 142c1bcd8 )  . 
 Added TimetableSnapshot to RoutingContext for improved route planning
 Fixed timeout calculation to prevent shutdown after short pings . 
 Fixed issue with processing canceled files . 
 Added import for Usernames in StubZMessagingApiStore
 Added dimension check for correct dimension registration in NetworkDispatcher . 
Added hasLineOfSight method to LivingEntity class
 Changed catalog URI in TestCatalogService to point to a different location . 
 Added new thread factory for Executors in CUDAContext for improved performance
 Added new enumeration values for EMERGENCY ,  HEALTHCARE ,  TRANSPORTATION ,  BARRIER ,  TOURISM ,  ENTERTAINMENT ,  and HISTORIC . 
 Added durable_writes attribute to CliClient . 
 Added new preference for number of free downloads . 
 Fixed issue with progress drawable not being updated correctly . 
 Fixed bug in size (  )  method to return correct count of objects in indexed set . 
 Changed method `severity (  ) ` to return `nullable String` instead of `String` . 
 Added method `create ( FolderType ,  int ) ` to LocalStore for efficient folder creation . 
 Fixed exception message in TaskListSubActivity to improve error handling . 
 Added max idle time for Gradle daemons during test execution  ( ttl = 10000 ) 
 Fixed getClientId method to return the client ID instead of the client ID field . 
 Fixed layout issue in EmbeddedDrawerActivity
 Fixed issue with $HOME variable in GoEnvironmentGoPathModificationTracker . 
 Fixed null pointer exception in TvdbTools . title calculation . 
 Added degree information to AugmentGraphWithDegreesJob . 
 Commit message :  Added walkSpeed and bikeSpeed parameters to ProfileRouter for improved search performance . 
 Added new enum constants for ChangedFrom and ChangeType in CommentActions class . 
 Added new test for TargetAction with mocked Picasso and corrected expected exception message
 Fixed default value for getShowReleaseTime to exclude invalid values outside of 0 - 2359 range . 
 Added new `name` field to `ConfigurationMetadataMatchers` class . 
 Fixed type check for AbstractAioChannel in AioEventLoop . java
 Added support for partial graphs in ProjectCommand . 
 Update VoiceRouter class to use new default speed value . 
 Added method to create Datastore with optional database name
 Added explicit close of input method token to prevent soft keyboard coverage
 Added DISABLE_ASYNC_SAVEFRAME and ENABLE_ASYNC_SAVEFRAME hints to control saving behavior in PGraphics  (  # 6792ec0 )  . 
 Fixed bug in handling departure times for stops 2 - nStops ,  now using arrival times instead of dwell times for calculation of min / max times . 
 Fixes issue with authentication token timeout in GtasksLoginActivity
 Added queryIntentServicesAsUser (  )  method to retrieve installed accessibility services . 
 Added call to notifyVideoAvailable (  )  after onTune ( Uri )  to signal video availability . 
 Added method `outputs (  ) ` to `OutputListResponse` for better API compatibility . 
 Changed method `hasWeighting (  ) ` to include a check for `weighting  !  =  null` . 
 Fixed bug in show (  )  method to properly pass duration parameter . 
 Fixed formatting issue in SitePickerAdapter . java ,  improved readability by using getBlogNameOrHomeURL (  )  instead of isRecentlyUsed  +  lastPickedTimeStamp  +  site . getBlogNameOrHomeURL (  ) 
Added getFragment (  )  method to SubPlan class
 Added serializable implementation for DefaultOAuth2ClientContext
 Fixed bug in WindowManagerService . java by removing unnecessary animating flag and updated hasStatusBarPermission check . 
 Fixed exception message for TreeWalker not being allowed as a parent of a Check . 
 Fixed exception handling in ConnectionHandler to prevent ClosedByInterruptException . 
 Added method to contain statistics about a database
 Changed method signature in `DefaultSerializerProvider . java` to include a `JsonMappingException` catch block to handle exceptions thrown by the `findExplicitUntypedSerializer` method . 
 Fixed bug in JSON serialization of objects with circular references . 
 Fixed license year in HiddenHttpMethodFilter . java
 Fixed compatibility issue with Java 6 by adding a check and throwing a GradleException if it ' s not met . 
 Added new test case for NBC Live availability through Yahoo Chat Web site . 
 Added `longValue (  ) ` method to `MapMembers` class for better performance . 
 Fixed exception message for indices created on or after 2 . 0 to correctly handle legacy routing settings . 
 Fixed warning message in AgentRegistrationElf . java to be more informative . 
Added embedded storage support for OMVRBTreePersistent . 
 Fixed method signature to match updated HDFS API in version 1 . 0 . 0
 Added log statement for action bar ID in Sherlock compatibility mode . 
 Added new ShapedOreRecipe class to replace vanilla recipes . 
 Changed the `MappingEntry` interface to remove unused imports and methods ,  simplifying the code . 
 Added constructor for VersionAck class
 Changed how the `BufferedReader` is constructed based on the `contentEncoding` parameter . 
 Changed how `andExpect` method is called to use `value (  ) ` instead of `equalTo (  ) ` for `jsonPath` to match the new API . 
 Added boolean safeIdBuffering to NeoStoreDataSource to disable buffering id generation
 Fixed issue HHH - 6337 by adding missing attribute bindings for joins . 
 Fixed indentation and removed unnecessary lines in ExampleTagBase . java
 Added always - same - presenter functionality to SinglePresenterSelector
 Added visibility check for watched button in MovieDetailsFragment
 Fixed bug in background image drawing . 
 Changed how the `search (  ) ` method handles the german query to only search the first field . 
 Fixed vertex shader loading issue
 Commit message :  Changed the way transactions are stored in the block tree to make it more efficient . 
```
 Added assertions for equality comparison of parameter descriptions
 Fixed issue with missing plugins . xml file ,  now using default configuration . 
 Added ForgeEventFactory to calculate break speed for item toss events
 Modified CompilerOptionsPreprocessor to require  ' remove_unused_prototype_props '  to be turned on for  ' remove_unused_prototype_properties_in_externs '  to work . 
 Added silent and vibrate settings to SettingsHelper class
 Added check for null view in bindView (  )  to prevent IllegalArgumentException
 Fixed bug in NettyProducer to correctly handle reusing channels when configured to do so . 
 Commit message :  Added import statement for Google Guava ' s Objects class in BlockInfoEntry class . 
 Added support for retrieving invoices with items
 Added default database location property to TestEnterpriseDatabase . java
 Fixed issue with incorrect response status code in WXSDKInstance . java
 Added config and disable mod buttons to GuiModList . 
 Added logging for post requests with session data . 
 Fixed issue with duplicate close (  )  calls in TxMakerTest
 Fixed issue in FileUploader . java to ensure file uploads only occur once . 
 Rolled back to savepoint due to max parallelism mismatch between savepoint state and new program . 
 Fixed issue with log statements in DefaultPackageScanClassResolver to improve trace output
 Added candidate and candidate score fields to WifiConfiguration class
 Added method to return edge weights in DirectedMultiGraph
 Fixed issue with overdue notification queue and refresh state for bundle . 
 Fixed return type of ResultSet in PreparedStatementProxy to match updated proxy method signature
 Fixed bug in keyguardIsShowingTq (  )  method
 Changed how `standaloneXmlTest (  ) ` is invoked in `ParseAndMarshalModelsTestCase` . 
 Fixed issue with directory selection in BrowseForDatabaseActionListener
 Fixed bug causing save session commit to be skipped in some cases . 
 Added support for parsing job data in ZooKeeperMasterModel . 
 Fixed exception message in Facebook . java
Added validation error handling for DeepLearning model . 
 Fixed timestamp and date types in ArrayTypeHandlerTest
Fixed a bug in the `AbstractEndPoint` class by updating the `hashCode (  ) ` method to include the connection hash code when null  (  # 8b3303579e )  . 
 Added static variables for CommandLineHelper to improve performance
 Added check for CANONICAL_SELF parameter in PythonDocumentationProvider
 Fixed timer scheduling issue in PeerGroup . java
 Added method to return class loader for resource package in ResourcesManager  ( b / core / java / android / app / ResourcesManager . java )  . 
 Fixed copyright year in ClassPathJaxb2TypeScanner . java
 Fixed bug in transit stop filtering to correctly use only nearby stops . 
 Commit message :  Added support for updated AppEngine SDK in the facet provider . 
 Added 50ms delay before updating SDP records to improve dbus stability . 
 Fixed bug in showGroupNotificationForActiveNotificationsMap (  )  to properly handle missing icon and title parameters
 Fixed issue with parsing color values in ImporterDOT class . 
 Fixed issue with scaling in AffineTransform class
 Added UndoManager to AceEditorNative for improved editing functionality . 
 Added support for displaying language dialects in PsiViewerDialog
 Fixed build target resolution for FullyQualifiedTarget
 Added check for invalid number of codewords in Aztec decoder
 Fixes package name validation to allow leading  '  /  '  and improves error message for invalid package names . 
 Fixed issue with negative maxLifetime value and added validation for maximum allowed value . 
 Fixed column layout updated to use new `span2` class for improved responsiveness . 
Fixed issue where autoInstallOSGiBundles was disabled due to missing contexts folder in OSGiAppProvider . java . 
 Added support for DUN APN type in CdmaDataConnectionTracker
 Fixed license header to comply with Apache License 2 . 0
 Added CodegenReflectionWrapper for improved performance
 Added case 3 for onPostAction listener to handle POST_COMMENT action
 Added method `ODocument ( Map < Object ,  Object >  iFieldMap ) ` to fill a document with field values . 
 Changed the Edge class to include a new `PlainStreetEdge` field and added a comment to log the geometry of the edge when it is serialized . 
 Added PersistentHashMap for GenericCompilerCache with new PersistentStateDataExternalizer for GenericCompiler . 
 Added new strategy options for experimental persistent Java compiler . 
 Added  @ Deprecated annotation to method isErrorEnabled (  )  and delegated functionality to org . slf4j . Logger # isErrorEnabled (  ) 
 Commit message :  Fixes CAMEL - 6414 by adding a missing closing brace in the `SimpleParserExpressionTest` class . 
 Changed ItemlistFragment to handle IllegalArgumentException when unregistering contentUpdate receiver . 
 Fixed bug in HdfsLoader class to properly handle missing external jar file
 Added method  ' archiveTreeMap '  to Pump class
 InstalledPackagesPanel :  added  ' Install '  and  ' Uninstall '  buttons . 
 Added summary for save current track preference
 Added workaround for incorrect TO attribute in IQAuthHandler . 
 Added try - catch block to handle IllegalArgumentException in findResource method
 Fixed bug in Environment . isDevelopmentRun (  )  to only consider Environment . DEVELOPMENT ,  rather than Environment . TEST . 
 Fixed color shift for Darcula theme . 
 Fixed bug in output file name generation . 
 Fixed null pointer check in Utility . java
 Fixed catch block to handle exceptions more robustly
 Commit message :  Updated FlickrPhotoGrid to use ImageVideoBitmapLoadFactory for loading bitmaps . 
 Changed behavior of `IdeNotificationArea` to show icon even when tool stripes are hidden or presentation mode is enabled . 
 Fixed method signature for file expressions in SimpleLanguage . 
 Fixed issue with package culling for stopped packages . 
Added ability to set column name and updated nullable ,  unique ,  and ignore annotations in Column . java
 Added loadLibs (  )  call to ImApp . java to shorten delay after typing passphrase
 Added new method  ' query '  and updated  ' clear '  and  ' size '  methods in InfinispanConfiguration class
 Added context variable for TimePickerDialog
 Fixed rounding issue in IntegerDivider class
 Added migration logic for InternalPartitionServiceImpl
 Added check for STATIC mode in ErrorCheckerService
 Fixed bug in TerrainQuad . java by adjusting normal calculation for offset . 
 Added visibility check for GlassPanel in DefaultSearchableConfigurable . 
 Commit message :  Fixes issue with incorrect file URL creation in VfsUtil . java . 
 Added focusability to KeyguardHostView to prevent volume keys from being handled by fallback handler  ( bug 7546960 )  . 
 Fixed issue with Realm . deleteRealm (  )  not properly cleaning up test Realm . 
 Added table header visibility control . 
 RescopeGlobalSymbols :  added new variable declaration
 Fixed issue with list scrolling in ReaderPostListFragment
Added import statements for PyFromImportStatement and PyImportElement classes in PyDocReferenceExpression . 
 Added test for detecting foreign keys to external schemas that don ' t appear in the changelog
 Fixed issue where node was being signaled as DEAD prematurely ,  updated retry count and last updated timestamp . 
 Changed test to ignore expensive test that is not related to Helios . 
 Changed behavior of IOUtils . canUseSnappy to use Snappy compression only if System . getProperty ( idea . no . snappy )  is not set . 
 Changed how StringUtils . processMutliLineSQL handles comments to remove them . 
 Added method to retrieve StringManager for a given class or package ,  reusing existing manager if possible . 
 Fixed bug in ContributionListing class to correctly handle duplicate contributions . 
 Added support for caching filters in FilteredQueryParser
 Fixed issue with null linked type in ORecordSerializerCSVAbstract . 
 Added new verifyAirplaneMode (  )  method for correctExtrasOnReceiveAirplaneDispatches (  )  test
 Fixed primary host removal logic in DefaultMultiServerCluster . 
 Added support for parsing the number of headers in the message
 Added getLifetime (  )  method to CoreModuleManager for better module management
 Fixed issue with null view parameter in getNumberOfMatches method
 Added import for com . raizlabs . android . dbflow . sql . language . Select
 Added default power SSID to ConnectivityManagerTestActivity
 Fixed formatting issue in BikeParkUnlinked class . 
 Added flag for retrying started service if original attempt failed
 Fixed buggy logic in TargetElementUtil . java to correctly handle references to attributes and names . 
 Fixed buggy code in AccessibilityNodeInfo . java
 Added new attachment mechanism for stateful information in ChannelHandlerContext . 
 Added import for org . tmatesoft . sqljet . core to handle SQLJet errors
 Renamed default parameters in Es6HandleDefaultParameters . java
 Fixed issue with popup window persistence in FMLEmbeddingRelauncher . 
 Added ourOpenFilesSetModificationCount increment in FileEditorManagerImpl . 
 Added check for single account mode and remote search support in manual search
 Fixed return type of `execute` method to match the actual implementation . 
 Changed method `exists` to accept an array of keys instead of a single key . 
 Added internal API for retaining custom non - configuration instance
 Added new Report Fields and updated existing ones in ConfigurationBuilder . 
 Fixed issue with pre - cached shares in database . 
 Fixed missing bean error message in AuthenticationManagerFactoryBean . 
 Fixed null pointer exception in TextEditorBackgroundHighlighter . 
 Changed the `stepic_id` field from optional to required in `Task` class . 
 Added method to set print logger in Ln class
 Commit message :  Fixes issue with deferred cancel process not being called after dragging item is dropped . 
 Fixed HystrixPlugins reset in test method
 Added  ' effectAllowed '  property to DataTransfer object for dragged tabs to allow for move action . 
 Added cancel button text and tooltip text to Task class . 
 Fixed lenient mode to handle invalid lines correctly . 
 Added import for ChannelPipeline in HttpClientCodec
 Added toPath (  )  method to File class in BaseCmd class
 Added import for org . jetbrains . jet . lang . resolve . scopes . JetScope and modified toString (  )  method to include scope information
 Added interface method for privacy list events . 
 Added support for transform inheritance in ModelInstance constructor
 Added support for iterable elements to be instances of Seq
 Added flag  - g to installFlags for granting runtime permissions
 Added delegate to FileSystem interface for long - term solution
 Added new mode for filling and stroking geometry and text
 Fixed bug in processing request API 5s ,  now using  % s instead of  % s . 
 Fixes font color calculation for null style . 
 Added support for starting invalid main JARs
 Added license text for packageDependencies
 Added progress bar to complete state UI
 Fixed bug in handling qualified references in package completion
 Added ability to wait for data to arrive before returning results . 
Added CONTEXT_FACTORY and EXTRA_ENV variables to DefaultLdapAuthoritiesPopulatorTests for better testing performance  ( commit f6f381dce .  . 25d11241b )  . 
 Added method `toString (  ) ` to Suppliers class
 Added method findClosest ( double lat ,  double lon ,  EdgeFilter edgeFilter )  to LocationIndex for efficient querying of closest nodes and edges . 
 Fixed inconsistent reference issue in ConsistencyCheck . java by updating prev . getNextProp (  )  to match property . getId (  ) 
 Fixed Guice error message to include more detail . 
 Fixed bug in IntervalTree implementation by updating results variable correctly . 
 Changed behavior of closing clients in `close (  ) ` method . 
 Fixed null pointer exceptions in Edge . writeObject (  ) 
 Fixed TLS negotiation logic in PacketReader and XMPPConnection classes . 
 Commit message :  Fixes  # 123456 :  Updated TabText and RadioButton text for better readability . 
 Added chunk boundary tracking for SwiftInputStream
 Added import for Google Guava ' s HttpHeaders and MediaType class
 Added static import for Groovy . groovy . Groovy . ratpack
 Added new setting for default zoom density in WebSettings class . 
 Fixed issue with deadlocks when stopping brute force scans
 Changed method `equals` from generic to non - generic . 
 Refreshing VirtualFileManagerImpl :  updated asynchronous refresh counter from 0 to 1 . 
 Added optimization for font rendering performance . 
 Fixed precision issue in TestSVMLightRecordFactory
 Changed how `Credential` objects are passed to `createTicketGrantingTicket (  ) ` to prevent passing null credentials . 
 Added handling for mouse clicks on the second button to trigger the hidden side action
 Added ZK root property to ZkUtils
 Added check for hiding windows when keyguard is active and not animating away  ( r3305e1e )  . 
 Fixes authentication issue and starts synchronization jobs on activation  (  # 884644e )  . 
 Fixed issue with annotation indexing in AnnotationUtilEx . 
 Added import for Apache HttpClient utility methods
 Commit message :  Fixes issue with incorrect data type handling in JDBCInputFormat . 
 Fixed method signature for NodeMulticastListener . isValidJoinMessage (  ) 
 Added / updated HTML ,  HTML5 ,  and XHTML template names . 
 Fixed bug in QueryImpl implementation to properly handle mappedValue for op . val (  ) 
 Fixed warning for excessive whitespace in token checks in PatternRuleTest . 
FAB :  Fixes issue with hasLollipopApi (  )  method in FloatingActionButton . java  (  # 426b2c2 )  . 
 Changed the `ServerTestUtils . writePropertyToFile (  ) ` calls to use the correct property keys for the `HighlyAvailableGraphDatabase` configuration . 
 Added new method to mock directory service to log index files . 
 Added alias provider for dependency labels in RuleContext . 
 Changed method signature to add new parameter for reporting window state changes . 
 Added check for too large files in GlobalInspectionContextImpl . 
 Fixed exception handling in ShellBolt to properly handle errors and exit the process . 
 Added COLUMN_INDEX_E164 and COLUMN_INDEX_STRIPPED constants for improved performance
 Added catch block for VirtualMachineError and Throwable in NioEndpoint . java
 Fixed issue with dx calculation in FrameData . java
 Added more detail display option to OsmandMapTileView
 Fixed issue with relative file paths in PushMojo class . 
 Added enable detection of mismatches between defined resource types and getter calls to StrictMode . 
 Changed `import com . intellij . openapi . fileTypes . PlainTextFileType ; ` to `import com . intellij . openapi . fileTypes . TextFileType ; `
 Fixed SQLException handling in ConnectionPool . java
 Fixed calculation of days to hours conversion in DebtConvertor
 Added support for tracking all Maven projects in the IDEA project file index ,  including those not located in the root directory  (  # 11 )  . 
 Fixed expiration policy for OAuth access tokens
 Fixed convention property assignment in ConventionProperty . java to prevent unnecessary warnings
 Added support for processing packets with NoResponseException in XMPPConnection . java . 
 Fixed exception handling for Twitter provider
 Fixed issue with verbose mode not being honored in HTTPServer constructor
 Added version to sstable filenames and updated metadata components in preparation for 0 . 7 . 0 release . 
 Added support for Roster Entry in packet . 
 Added import for HttpHeaders in QueryResource
 Added implementation of Funnel interface for common types in Funnels class . 
 Fixed buggy code in AbstractSctpChannelSink . java
 Added `updateListValid` field to Node clone to prevent unnecessary updates . 
 Added hook update test with incremental value
 Changed method signature from `ExecutorService` to `ScheduledExecutorService` in `ReplicatedMapConfigReadOnly` . 
 Changed the JmsConfiguration class to include a new transacted field ,  and added a test to verify its value in the JmsEndpointConfigurationTest class . 
 Added feature to retrieve groups for candidate user  (  # 7cfa61f67 )  . 
 Fixed issue with removing of network connections though connection is active . 
 Update to WordPress . java to include rate limited WPCom blog list update
 Fixed issue with backing up index with no commits
 Added configuration for Tomcat embedding
 Added check for available capacity being less than 1 in Sequencer . java
 Added new method `set` to `SmartList` to replace old `set` method
 Fixed bug where endTime was returning incorrect value for task finished events . 
 Added proxy credentials to ApacheAsyncHttpProvider for authenticated requests
 Added field  ' isFavorite '  and method  ' isFavorite (  )  '  to Show class . 
 Fixed issue with color selection in GPX fragment . 
 Added visibility check for discover data in ReaderPostAdapter . 
 Changed the `setTrace` method to `setBuildParseTree` in `TestRig . java` to improve performance . 
 Added formatting to StoredClientChannel . closeStr for better readability . 
 Added method createNativeQueryTyped (  )  in QueryHandler
 Added new method to remove cross - validation predictions in ModelBuilder
 Added set . setDetachWallpaper ( true )  to correctly position upper and lower wallpaper targets . 
 Added import for Build version in Snackbar layout
 Fixed method signature in Location . java to match Bukkit API
 Fixed NullPointerException in Request . java by updating null check to `if  ( asyncContext  =  =  null ) ` instead of `if  ( asyncContext  =  =  null || asyncContext . isStarted (  )  ) ` . 
 Changed the definition of SIM_BUSY ,  SIM_FULL ,  NETWORK_REJECT ,  OPERATION_NOT_ALLOWED ,  and EMPTY_RECORD constants . 
 Added new test cases for DataServerIntegrationTest
 Fixed issue with parent element retrieval in AbstractListBuilder . 
 Added ability to edit commit message in VcsHistoryDialog . 
 Changed how the IllegalArgumentException is handled in `PreBuiltTransportClientTests` . 
 Fixed bug in DalvikBitmapFactory to correctly erase transparent pixels
 Fixed bug in close method to properly trim reason length
 Added support for handling throwable types in TypeDescription . 
 Fixed bug in Parser . java by updating _parseListeners field . 
 Added exception handling for znode creation in LlapZookeeperRegistryImpl . 
 Added new admin receiver to DevicePolicyManagerService . 
 Changed method `restoreFromBackup` to use `singleThreaded  =  true` for improved performance . 
 Added new methods for generating REST files in the `ImportFiles2` and `Parse2` classes
 Added  ' reference '  theme to ResReferenceValue constructor
 Fixed bug where findFirstVisibleItemPosition (  )  was returning incorrect value ,  now correctly returns the position of the first visible item . 
 Added support for varDataEncoding . length field in ExampleUsingGeneratedStub . java
 Update query filter to use ColumnFamilyStore . removeDeleted for faster index building
 Fixed log statement to include the error message for better debugging . 
 Changed how `ElasticsearchParseException` is thrown when loading a null - valued setting . 
 Fixed bug in NfcBarcode . java to support Thinfilm NFC Barcode tag specification  ( previously Kovio NFC Barcode )  and updated documentation links . 
 Fixed issue with starting activity in SearchDialog ,  now correctly handling explicit component . 
 Added extra default exception mappings to ProviderManager . 
 Fixes signature of deprecated method  ' buildUnionSubQuery '  to avoid confusion
 Fixed obsolete version exception in BdbStorageEngine for key   +  new String ( hexCodec . encode ( key . get (  )  )  )   +   .  Current version is   +  value . getVersion (  )  . toString (  )   +   .  Committed with 100644 . 
 Fixed final class to make it non - final
 Fixed URL in ActFmInvoker to point to production API . 
 Changed repeatWhen (  )  to retryWhen (  )  and updated documentation links . 
 Added overrideConfig for hbase . zookeeper . znode . parent and hbase . zookeeper . quorum to TSDB constructor
 Added check for double - spent transactions in Wallet . java
 Fixed bug in TransitIndexBuilder to properly handle missing departure / arrival times
 Added toString (  )  method for HiveType class
 Fixed case HYBRIDHASH_BUILD_FIRST_CACHED and added new case HYBRIDHASH_BUILD_SECOND_CACHED in PlanJSONDumpGenerator . 
 Added null check for mSessionCallback in constructor
 Added new lucene query generation functionality to MatchFunction
 Fix backpressure buffer implementation in OperatorOnBackpressureBufferTest
 Added support for ES6 `super` keyword conversion
 Added / removed menu items for ExtensionLoader
 Fixed formatting issues in ServerListDownloader . java
 Changed to use ScrollableSingleRowLayout instead of CompressibleSingleRowLayout . 
 Added ability to create InputCreated instances with a non - empty ID . 
 Added support for IncrementalBackupNotPossibleException in BackupTool . 
 Added check for restricted UID in PrivacyManager
 Added option to set the block size for key value partition files
 Added method to return View in RemoteImageLoaderHandler
 Fixed scaling issue in BoundingBox . java by using abs (  )  to calculate extents . 
 Fixed empty variable check in HiveSchemaHelper for correct parsing
 Added  ' OK '  button to PreferencesFrame
 Added shared memory directory props for Linux to avoid Disk I / O bottleneck in AdminSoakTestRunner
 Fixed issue with processing fields with property names
 Fixed bean info conversion for body parameter type . 
 Added BladeWebContext . setContext (  )  method for setting the ServletContext in DispatcherServlet . 
 Fixed issue with long - running content loading thread . 
 Fixed issue with Vagrant instance being down . 
 Fixed handling of failure messages to use byte [  ]  instead of Throwable . 
 Fixed compiler invocation to use newInstance (  )  method instead of direct instance creation . 
Added support for EJB client context processing in deployments containing jboss - ejb - client . xml
 Added synchronized flush and close methods to ensure thread safety
 Added check for RemoteTransportException cause in DistributedMergeTask . java
 Fixed formatting issue for ISO period  (  # 93 ) 
 Upgraded LwjglGraphics class to use GL 2 . 1 for better performance on some platforms . 
 Fixed cancelation mechanism in AsyncHttpClient to prevent leaks
 Added support for updating row count for BIGINT columns
 Added static methods for rotateY (  - 90 )  and rotateY ( 90 )  in BlockTransformExtentTest . 
 Fixed bug in PSurfaceJOGL . java :  corrected incorrect line number in pgl . setCaps (  ) 
 Added trimV4AddrZeros method to trim IPv4 addresses of leading zeros
 Fixed null pointer exception in Model . java by replacing null check with validation . 
 Changed how script execution is handled to prevent null stream exceptions . 
 Commit message :  Update BatchLayer and SpeedLayer classes to use specific versions of Spark and YARN dependencies . 
 Added import for Constants class
 Added new attribute mapping to MemberImpl
 Changed the `Windowed` class to implement `Equals` and `HashCode` for better performance . 
 Fixed view type count check in Spinner adapter for API levels L and above . 
 Changed type matcher for AnnotationTypeMatcher class . 
 Fixed bug in VcsException handling
 Fixed NPE in `bar (  ) ` method in `NPEOnAnyClassMatcherAutounboxTest` to match issue  # 221 . 
 Added test tier with single storage dir under test directory . 
 Fixed getConfigurationName method to return the correct name for the current configuration  (  # 6a2bccc )  . 
 Fixed issue with deprecated method in TileService ,  now using new approach for fetching travel time tiles . 
 Changed implementation of `JavaMethodCallElement` to include location information in the presentation . 
 Commit message :  Added support for iterating over nested collections in OCommandExecutorSQLSelect . 
 Fixed issue with incorrect import statements in Util . java
 Fixed issue with embedded Maps in PersistedServiceImpl by converting them to new HashMap instances . 
 Added exclusive lock acquisition for cluster creation
 Added  ' org . hibernate . lockMode '  and  ' org . hibernate . flushMode '  query hints to AvailableSettings class
 Changed how `VisiblePack` is initialized to handle nullable visibility packs . 
 Added method getStyleAttribute_doesNotThrowException_whenStyleIsBogus (  )  to TestAttributeSetTest
 Added initialization for BlockMasterClientService . Client in RetryHandlingBlockMasterClient
 Resolved build rule resolution issues in BuildRuleResolver . 
 Added serialized subscriber for OperatorTakeUntil to handle multiple observers
 Added new line of code to handle matchedEOF
 Added constructor for JedisPool to accept a string host parameter . 
 Fixed registration of key bindings to use correct field name . 
 Fixed timeout exception in TestSqlStageExecution . java to match updated behavior . 
 Fixed deprecated method in Rule . java to align with Sonar 3 . 6 API changes . 
 Added getResourceConfig method to JerseyEnvironment
Added support for Broadcaster scope in BaseTest . 
 Fixed type argument for  ' Log '  method to avoid ClassCastException
 Fixed issue with invalid config values in PaneManager . 
 Added method to handle failed computations in Future interface
 Added todo to assert rawquery with args returns actual values  (  # 335 ) 
 Added setupDateTimeInterpreter (  )  method to configure date and time formatting in MainActivity
 Changed handling of mouse wheel events to properly account for negative values on OS X with natural scrolling enabled . 
 Fixed issue with unknown pipe key detection
 Added backwards compatibility hack for MultiMessageCodec interface in DecodingProcessor
 Added support for Elasticsearch operation  ' EXISTS '  in ElasticsearchConfiguration . 
Added commit logic for reverting state in VariableInplaceRenamer ,  including updating the top - level editor and committing the document . 
 Changed the ignore list to add new URLs and remove old ones . 
 Added debug level variable to Graphs class
 Changed the way Class . forName is called to improve performance . 
 Added default max capacity to Recycler class
 Added method to clean up file paths in ChangedFile class
 Added missing import for water . fvec . Frame
 Changed behavior of `useOutOfProcessBuild (  ) ` to only consider compilers that are actually used in the build process . 
 Added support for parallel two - phase aggregations in Aggregator class
 Fixed bug in SessionImpl . java by adding a finally block to catch and handle exceptions in the checkNoUnresolvedActionsAfterOperation method . 
 Fixed bug in EventOriginClientPropagationPolicy by updating nullable EventOrigin annotation to non - nullable
 Fixed null pointer exception in FallbackTypeConverter . unmarshaller (  )  by creating a new instance . 
 Added new test case for non - greedy termination and optional grammar rule
 Fixed issue with inflected elements not being properly handled in the rule . 
 Added support for loading Grails build listeners via build settings . 
 Fixed issue with missing blob cluster IDs in ODatabaseImport . java . 
 Changed how `CustomProperty . ORIGINAL_ID . setValueUnlessPropertyExists (  ) ` method is called in `testSetValueUnlessPropertyExists (  ) ` method . 
 Fixed serialization exception in VoldemortClientShell class . 
 Changed implementation of `DataPack` class to use a new `GraphFacade` instance instead of a custom `GraphFacadeImpl` . 
 Added session management for AtmosphereRequest . 
 Added support for color 99  ( Transparent )  in IRC message processing . 
 Fixed case sensitivity for JSP search . 
 Added logging for interrupted exceptions in NetworkInstance . 
 Added support for client certificate vector size in Settings . 
 Changed how the scanner is created in `postFlow` to use the `context . get ( Scanner . class ) ` method instead of `context . get ( TreePathScanner . class ) ` . 
 Refactored FeedRefreshWorker class to remove unnecessary transaction management annotation
 Fixed null return in decode method to prevent IllegalStateException when no data is consumed and state didn ' t change . 
 Added import for Intent to handle activity results in BillingActivity
 Fixes issue with incorrect column offset calculation in UnicodeTranscript . java by updating findStartOfColumn method . 
 Changed implementation of reflection mechanism for resolving properties / methods / fields in StandardEvaluationContext . 
 Added drag - n - drop selection functionality to UI_GraphTable
 Added getWidth (  )  and getHeight (  )  methods to QRCodeEncoder class for easier access to image dimensions . 
 Fixed bug in SystemProperties . java by adding missing license header . 
 Added read listener for WebSocket upgrade
 Added support for Plurk ' s new OAuth 2 . 0 authorization flow
 Fixed compile error in ByteBuddyTest by adding missing enum values . 
 Added method to pause and quit the draw thread in CacheManagingDrawTask
 Changed how the file is written to the client ,  now using a `MessageList` instead of hardcoding the file size . 
 Added check for null packet before sending . 
 Fixed class names in OrientJdbcDatabaseMetaData to match metadata changes . 
 Fixed initialization of vanilla entries . 
 Added support for Android Support Annotations
 Added sender ID to GCMIntentService constructor
 Added cleanup code for popup factory recycling in LookupImpl  (  # 81c830 )  . 
 Added setCenter (  )  method to CylinderRegion class
 Added source active document with echo functionality in TextFileType . 
 Added HIDE_PLACEHOLDER_PIC property to OneToOneCallPeerPanel . 
 Update to use new WordPress API endpoint for getuserblogs . php
 Fixed issue with password encryption in InMemoryUserDetailsManager
 Fixed issue with null pointer exception in ErrorCheckerService . 
 Fixed bug in ChannelUpstreamHandler to properly handle interestOps changes . 
 Added invalidateParentIfNeededAndWasQuickRejected (  )  method call in View . java for hardware acceleration optimization . 
 Changed test to include thread safety test for constructor expression . 
 Fixed nullability checks in Moco . java
 Added support for splitKeyMin and splitKeyMax in StandaloneMongoSplitter . 
Added support for highlighting braces in injected languages . 
 Changed how pedestrian street names are generated . 
 Inlined variables and aliases in InlineVariables . java ,  and added test for no inline exported name2 . 
 Added configuration option to allow fallback to simple authentication for IPC client
 Fixed indentation and added missing closing brace in ErrorBar . java
 Fixes bug pattern  ' WaitNotInLoop '  by replacing if statements with while loops . 
 Fixes ShadowShapeDrawable to extend ShadowDrawable
 Added method `numGetMores (  ) ` and deprecated `getSizes (  ) ` in DBCursor . 
 Changed how the `Preconditions . checkState` and `Preconditions . checkArgument` methods handle null arguments . 
 Added log warning for restricted IPC calls in XBinder
 Changed visibility of menu items for FragmentTuner based on action state . 
 Fixed issue with IndexOperator . EQ comparison in AvroValidation . java
 Fixed whitespace issue in PApplet . java
 Fixed issue with cache word uninitialized message . 
 Added missing import for com . intellij . openapi . application . ApplicationManager in ModuleType . java
 Fixed bug in ByteBuffer allocation
 Fixed SQLException handling to use errorHandler . handleException (  )  instead of throwing a SystemException . 
 Fixed issue with Show History command not working on Mac desktop ,  added new Todos for Cmd - W close window and testing on all platforms . 
 Added method to retrieve views from parent view and fixed exception handling . 
 Fixed issue with test discovery for TestNG ,  now correctly handling module - scoped settings . 
 Added new fields to MongoOptions class :  maxAutoConnectRetryTime ,  slaveOk ,  readPreference ,  writeConcern ,  safe ,  and w . 
 Added default maxKeys value to ObjectListing constructor
 Fixed scaling mode for video renderer to improve performance . 
 Fixed bug in SubActivity2 . java by adding missing import statement
 Fixed Java files directory in KotlinMultiFileTestWithJava . 
 Added cancelTemplate (  )  method with  @ deprecated annotation
 Fixed bug in ScreenInformation . java by updating Point left and right coordinates to match the correct window position . 
 Changed DataType class to include columnSizeUnit in the output . 
 Added support for GLFW on Mac ,  Windows ,  and Linux platforms . 
 Fixed bug in downSampleSceneInto1x1pixelsBuffer (  )  to correctly handle exposure update . 
 Fixed bug to allow proxy script URL to be set when port is non - zero . 
 Added import for Utility class and updated getEnableSound (  )  method to also check for system ringer mode . 
 Added constructor for DBCollection
 Fixed class loading issue in Weaver class
 Added ability to compact rows with tokens in ranges . 
 Fixed issue with idleTimeout being close to or greater than maxLifetime ,  disabling it . 
 Fixed unsafe mode issue in DataStorePerfTest . 
 Disabled Jar URL connection caching for memory leak prevention . 
 Changed implementation of `updateArcFeatureStatus` to include check for `HdmiDeviceInfo` availability before activating ARC . 
 Fixed bug in HideUntilControlSet . java by updating the message for when the value is not 0 . 
 Changed output message for `jobs` command to correctly reflect the updated behavior of the `hostPattern` variable . 
 Added support for generating a unique ID for Signavio artifacts
 Added notifyViewAccessibilityStateChangedIfNeeded (  )  for temporary detach . 
 Fixed loadShaderProgram (  )  method to load correct shader programs
 Added map for storing source place of modules in EclipseModuleManager
 Added method isBodyEmpty (  )  to HttpRequest class
 Added method  ' copyMemory '  to UnsafeUtil for copying bytes from source to destination address . 
 Fixed path for tool in AndroidSdk class . 
 Added _curator . start (  )  method to ZkCoordinator class
 Added new condition variable for not - empty checking in TaskQueue . 
 Added gradient accumulation for GLMMultinomialGradientTask
 Added support for injected language injections in LiteralExpressionTokenizer . 
 Added new height and width parameters to Style class . 
`Added handling for managed profiles in DevicePolicyManagerService`
 Added default timeout for connecting to remote database instances
Added method isCommentComplete ( comment ,  commenter ,  editor )  to EnterHandler for improved comment completion functionality . 
 Fixed indeterminate progress bar visibility in LocalIndexesFragment . 
 Added isValid (  )  method to Phone class with normalization check
 Added PeopleListFilter class for improved filter functionality
 Commit message :  Implemented test for escaping Javascript in JavaScriptTagLibTests . 
 Added support for animator auto - cancellation in MainActivity . 
 Added GENERATE_EXPORTS field to Protocol class
 Fixed issue with cursor position after moving selection to next line . 
 Changed method signature to return `ItemStack` instead of `void` . 
 Added local history actions to VcsQuickListPopupAction
 Fixed bug in GenerateUtils to handle missing domain classes . 
 Fixed issue with large queries in AbstractTestDistributedQueries . java
 Fixed deprecated method signature in HttpClient . setProvider ( String ) 
 Fixed size of SurfaceView updated in PluginFullScreenHolder . 
 Added ShadowRelativeLayoutParams class with improved layout rules
 Fixed issue with custom date filter not being applied correctly in MediaGridActivity
 Added  @ Required annotation to mark field as not nullable
 DeprecatedJsonReport :  updated with new deprecated method name
 Added method handler for websocket message type . 
 Fixed bug in record creation
 Added new example for success response with updated code . 
 Commit message :  Changed implementation of OrderInsensitiveTaskFilePropertyCompareStrategy to use LinkedHashSet for improved efficiency . 
 Fixed type safety issues in KinesisConsumer constructor
 Added check for application starting window . 
 Fixed todo comment in IndexDatabase . java
 Added commit node caching for improved performance . 
 Fixed touch events enabled / disabled logic in Wallpaper . java
 Added NodeIdFactory and UuidGenerator beans to AbstractCamelContextFactoryBean
 Fixed bug in material node implementation
 Added  @ ThreadSafe annotation to KeyValueMasterFactory for improved performance and correctness
 Added prefix to EventBus tags for improved logging and debugging
 Fixed method signature in InputCovariant . java
 Added method  ' addAllRows '  to QueryResultCollection for efficient collection of query results . 
 Added log statements to track newTop and topBound during onViewDragStateChange (  )  for SlidingUpPanelLayout . 
 Changed order of directories in DirectoryIndexTest to improve consistency . 
 Commit message :  Added support for DNS resolution in EurekaHttpClients . 
 Added new implementation class cache mechanism for faster resolution of interfaces . 
 Changed unnecessary parentheses in code to improve readability and avoid warnings . 
 Added ability to write to output stream directly without buffering
 Added ProxyIgnore annotation to ConnectionTerminationTestCase to ignore proxy settings
 Changed how default object descriptor is checked to match Kotlin language spec . 
 Added support for concatenating empty arrays . 
 Added check for inserted text mode before starting drag . 
Fixed type safety in HandlerMapping . java by adding  @ Publisher annotation  ( diff  - r 605676a .  . da3284e )  . 
 Changed how spendable outputs are added to test spending non - existent transactions . 
 Fixed robot creation to use GraphicsEnvironment instead of Toolkit . 
 Added method `partition ( Predicate <  ?  super T >  predicate ) ` to `List < T > ` .  ) 
Added support for logging to  ' httpd . log '  for fatal levels  (  # 273e068f6 )  . 
 Fixed exception handling for PaymentApiException with proper error message . 
 Changed loading and filtering of XML in `Main . java` to improve accuracy . 
 Added memoization for headersSupplier to improve performance
 Fixed log message formatting for mod  % s . 
 Fixed bug in status handling for FileDownloadStatus . 
 Changed how parameter names are compared to determine if they contain a specific property ,  improving performance and correctness . 
 Added LGPL v2 . 1 license to Restore . java
 Added support for multiple language options . 
 Added test for parsing support of no - value cookies
 Fixed validation for AccountId parameter in AmazonKinesisFirehoseToRedshiftSample and AmazonKinesisFirehoseToS3Sample classes . 
 Added new theme names :  Tomorrow Night Eighties ,  Tomorrow Night 80s ,  and Twilight . 
 Fixed type variable erasure in GenericTypeDescription . getSuperType (  ) 
 Fixes bug in background . java by removing unnecessary import
 Update to Groovy 1 . 8 . 0 - rc - 4 in GrailsCoreDependencies . 
Added support for embedded storage in ODistributedAbstractPlugin
 Fixed bug in GPX file handling by updating file name comparison to use lowercase and dot notation . 
Fixed parsing of Javadoc tags in AbstractJavadocCheck to correctly handle non - void METHOD_DEFs and CLASS_DEFs  (  # 15899870f )  . 
 Fixed bug where featured tags were not draggable
 Fixed validation code to use the correct MessageSource bean . 
 Fixed issue with invalid characters in OSchemaShared class . 
 Added default vertical scrollbar position handling for RTL layouts
 Added counter to prevent infinite message loop in Encoder . 
 Fixed layout for JBRunnerTabs to correctly display tabs when hideTabs is enabled
 Fixed boolean return value for NotificationSettings . isNotificationsEnabled (  )  to match the default value of true . 
 Added 5 thread executors for improved performance . 
 Fixed issue with temp directory deletion in BaseDeployer . 
 Added implementation for ActionNotFoundTransportException with action field . 
 Added NO - OP for preloading texture ,  as new texturing system in MC 1 . 5 no longer requires it . 
Fixed a bug in the `StringTemplateParser` class by adding a missing `}` character in the `else` block . 
 Added import for MessageDefinition class and updated createResponseMessage method to load message structure from results array . 
 Fixed exception message in MasterInfo . java to improve error handling . 
 Fixed import errors and reduced code size in Model . java
 Fixed issue with protected accounts in TweetFragment
 Fixed formatting issue in KafkaConsumer . java with  % n instead of  % n % s for correct line ending . 
 AbstractCacheMonitor :  updated to log exceptions
 Added import for TextUtils class to handle empty string checking in MobileSignalController . 
 Fixed exception message in SpiceManager . shouldStop (  )  for better error handling . 
 Fixed bug in background color drawing for thumbnails
 Added default layout params for vertical and horizontal layouts
 Changed how the response header is printed in the test . 
 Added method to calculate percentage cache misses and updated JMX metrics . 
 Added warning for loading multiple instances of DefaultNodeExtension class . 
 Added method `getOffset` to compute the offset of a variable - length field in the tuple . 
 Added method getLocalQueueStats (  )  to return service - created LocalQueueStats instead of null . 
 Added method to retrieve group memberships for a user
Fixed typos and updated copyright year in SpringBeanAutowiringSupport . java
 Fixed issue with isStructuralType (  )  method in FunctionType . java
 Fixed issue with incomplete drafts being marked as completed . 
 Fixed decodeReplies method to match the updated test case . 
 Added method  ' tryJoinRawLines ( Document ,  PsiFile ,  int ,  int )  '  to JoinRawLinesHandlerDelegate interface . 
Added import for GitHub token in ReleaseNotesBuilderFactory to handle it properly
 Fixed exception handling in ApiHelper . 
 Added method for clearing module cache in CompileEnvironment
 Added method to allow suspending of AtmosphereResource before adding it to Broadcaster
 Fixed issue with zero - length intervals in YAxisRenderer . 
 Changed the disabled plugins list in PluginsTest . java . 
 Fixed issue with incorrect module name in Eclipse classpath storage provider . 
 Added conversion for UserAge field in MyConverter class
 Fixed bug in animation timing calculation to ensure correct playback . 
 *  `TransportResponse . java` :  Added abstract class for TransportResponse
 Fixed bug in bootstrapMetadataWithRetries (  )  to correctly handle STORES_KEY store name
 Added support for  -  - all branch option in GitHistoryUtils . 
 Added support for long and Boolean types in Registry . 
 Added getHandledPredicate (  )  method to HandledPredicate class
 Added MinecraftServer object to FMLServerStartingEvent class
 Added Type class with Hindley - Milner style type inference  (  # 3465959b3  -  >   # 912e3a78d ) 
 Fixed clipping issue in StackScrollAlgorithm for transparent views . 
 Added ability to import jobs via the API using PojoUtils . copyProperties . 
 ViewRootImpl :  added fence call to prevent pending invokeFunctor messages from being processed after detachFunctor
 Changed inflater configuration for CalligraphyContextWrapper to use false for `mAttributeId`
 Added method to create storage directory path with proper permissions and sticky bit  (  # 168 - 27 ) 
 Added implicit transaction handling for TransactionalService
 Resized bitmap returned for WP image span
 Optimized Java lexer for search helper
 Added  @ Produces (  ' application / json '  )  to ResourceChat # suspend (  )  and updated  @ Broadcast to exclude writeEntity
 Added parameterized test for inherited role and user . 
 Fixed issue with black pixel detection in AndroidGraphicsGridSampler . 
 Added jar file to JobManagerITCase test
 Fixed exception handling in ResourceHttpRequestHandler to prevent unnecessary IOExceptions . 
 Fixed bug in runInternal variable visibility . 
 Added import for org . gradle . model . internal . core . Service in JvmComponentPlugin
 Fixed mod ID injection for ProxyInjector class
 Resolved import element implementation . 
 Added new method to replay decoder to handle replaying of packets
 Changed method signature in `ReflectionUtilTest` to match JUnit 5 conventions . 
 Added ability to disable condition check for breakpoints . 
 Fixed license text in BaseIndex . java
 Added theming support for Toolbar in Android 5 . 0 and later
 Added validation for arguments in GenericTypeOperationHandler . 
 Fixed return type of post method from Void to Void
 Changed priority field to be a valid range of  - 128 to 128 . 
 Added new configuration properties for idle timeout and minimum connections per server . 
 Added new insets calculations for RecentsView to handle system UI changes . 
 Added Registration class to ClientSerializerRegistrationsService
 Fixed buggy catch block in HttpConnection . java to prevent exceptions from being swallowed . 
Added setUVRange method to MeshBuilder for setting texture coordinate range  (  # 8 ) 
 Added ValueAnnotation for word tags in SemanticGraph . 
 Fixes buggy call to super method by adding missing  ' super .  ' 
 Committed changes tree browser updated to include new feature . 
 Fixed issue with choice value out of bounds in SetType . java by updating the check for value . longValue (  )  . 
 Added support for syntax table completion in WordCompletionContributor
 Added check for null sysroot value in CppConfiguration . 
 Added setMessageId (  )  and setInReplyTo (  )  methods to MimeMessage class
 Added visibility modifier to private method for testing purposes
 Fixed bug causing keepAliveTimeout to be canceled twice . 
 Fixed bug in handling unregistered URIs to prevent error popup from appearing repeatedly
 Added synchronized block to OperatorContext . run (  )  to prevent thread interference
 Added null node check and improved code organization
 Changed how input stream is closed to prevent potential exceptions . 
 Added constant check for non - sparse chunks
 Fixed bug in TachyonFile . java to correctly handle block index in file metadata
 Added new sample data to the list view in SimpleActivity
 Added preferred height check for single line mode and improved display of small heights for better layout . 
 Added method  ' setPairedInfo '  and modified related code to enable pairing configuration storage . 
 Added support for configurable date and time format in  % t or  % {format}t . 
 Fixed NPE in MediaServiceImpl . java by commenting out callSession . put (  )  and using activeCallSessions instead . 
 Fixed blank line issue in CallableMessageSender . java
 Update to QuoteOfTheMomentClient and QuoteOfTheMomentServer classes to match official Java tutorial examples . 
 Fixed issue with member accessibility checking in TrampolineCompiler . 
 Fixed visibility of  ' Add Cc / Bcc '  menu item in MessageCompose activity . 
 Commit message :  Fixes issue with rapid put / remove of preferences causing unexpected behavior . 
 Added check for javax . scripting availability and warning for Management Center scripts . 
 Added SearchEverywhere action to ActionToolbar for main toolbar and navigation bar . 
 Changed XMLLanguage and XMLParserDefinition imports to include LanguageParserDefinitions and added a new XMLChildRole provider . 
 Added support for custom XML root element names in ModelResolver
 Added logging statement for query ID in AsyncQueryForwardingServlet
 Fixed bug in AppWidgetServiceImpl . java by removing unnecessary uid check
 Added test suite for ImmutableSortedSetTest . 
 Fixed Java code formatting for improved readability . 
 Added getContextProperty method to retrieve context property values
 Changed left child indent in DeferredIconImpl to match UIManager setting . 
 Added new `hasLegacyIndexChanges` field and updated `logPosition` field in TransactionCommitment . 
 Fixed exception handling in DataServerMessage . java
 Fixed issue with malformed headers in ExtensionCompare . java
 Added absolute path to local warehouse directory in HiveZeppelinDriver
 Changed how warnings are asserted in `BaseCheckTestSupport` . 
 Fixed multiple nops before switch / array data pseudo instructions and replaced const - string / jumbo with const - string instructions when string index is too big . 
 Fixed issue with saving attachment file names
 Fixed deadlock issue in SquidDatabase . acquireExclusiveLock (  )  by preventing call while holding non - exclusive lock or in transaction . 
 Added support for serializing and factories for AccumulatorState in AbstractMinMaxAggregation
 Added check for invalid combination of script and doc in UpdateRequest
 Added new SHA - 1 hash for guava artifact in WORKSPACE file
 Added ROOT_ACCESS constant for domain admin access
 Added roamingType = Disabled to UsageStatisticsPersistenceComponent for improved performance . 
 Added call to startServer (  )  method in GraylogControl class to start the Graylog server before running integration tests . 
 Added new equality comparison method for Graph objects . 
 Fixed buggy implementation of PopupComponent in b / platform / platform - impl / src / com / intellij / ui / popup / PopupComponent . java
 Added new exception  ' Euch '  to CaseRule
 Added mapping for  ' json '  to  ' application / json '  . 
 Fixed null pointer exception in ImagePresenter . java by updating the  ' id '  and  ' streamOpener '  parameters in the  ' fetchImage '  method . 
 Added OMEMO fingerprint for Axolotl service
 Added test to verify recompute executor creates a recompute plan and launches the recompute job at heartbeat  (  # 13c8b4f ) 
 Flushed OS page cache on IOLimiter to prevent excessive IO work . 
 Changed the `TestUnion` callback implementation to print `f2` value and set `called [ 0 ] ` to true before calling `read (  ) ` . 
 Added compatibility check for BloomFilters . 
 Added calculation of real turn cost for PlainStreetEdge in b / otp - core / src / main / java / org / opentripplanner / routing / edgetype / PlainStreetEdge . java
 Fixed bug in measure creation for Cobertura parser
 Commit message :  Fixes issue with incorrect type inference in PostfixTemplateTestCase . 
 Added new method `getUnchecked (  ) ` to `LoadingCache` for consistent API with `Cache` . 
 Added field for self - call optimization in OperationThread
 Fixed method stub in TestStreetMatcher . java
 Fixed calculation of leaf nodes capacity to improve performance . 
 Fixed formatting issue in Chronometer . java
 Added method to return true if a doc was found to delete
 Fixed bug in multi - node mode for replicated data . 
 Added support for GridLayoutManager in UltimateRecyclerView . 
 Changed how the app name is retrieved from the package manager . 
 Added Pair < ATNConfigSet ,  ATNConfigSet >  return type for ParserATNSimulator # getPredictedTokens (  )  to match Java 11 +  API changes
 Fixed null pointer exception in Gdx2DPixmap constructor
 Added ability to skip incomplete GELF messages . 
TypeCheckTest . java :  added test for duplicate static property declaration  ( 7 lines ) 
 Added performance test for uploading functionality
 Fixed warning message for missing connected output in Wallet . java
GenerateSourcesMojo :  Fixes issue with BuildConfig generation for AAR dependencies . 
 Added delegate . transaction (  )  method to SingleEntityStoreFromBlocking
Added QueueConfig getQueueConfig (  )  method in Config . java
 Added prefix to operation string to prevent concurrent modification exceptions
 Added userPrincipal field to GlassFishRequestUpgradeStrategy class
 Fixed issue with empty extension list in Launcher . java
 Fixed formatting issue in FileSystemLogger . java
Fixed issue with incorrect calculation of block length in JavaGenerator . java
 Fixed layout for LookupCellRenderer to improve rendering performance  (  # 3454 )  . 
 Added retry mechanism for handle . invokeWithArguments (  )  to handle exceptions more gracefully . 
 Added import for com . github . dreamhead . moco . HttpMethod
 Fixed bug in MimeUtility . java to handle OutOfMemoryError correctly . 
 Fixed issue with null row handling in GDataImporter class . 
 Added method `appendToPathEnvVar (  ) ` to GeneralCommandLine class for appending a value to the end of a path - like environment variable using system - dependent path separator . 
 Update FileDownloadTask to reflect transfer status changes . 
 Commit message :  Renamed reduceByKey method to LongSumReducer and added join method to combine ipAddressesCumDStream with ipAddressesCountsDStream . 
 Added default edge label  ' d7 '  for yEd compatibility . 
 Added support for setting request proxy using the `proxy (  ) ` method . 
 Fixed null value handling in OtpsCsvOutput . java
 Added new panel extension to AbstractProjectSettingsStep
 Fixed exception handling for MBean Dumper . 
 Changed method `createFile` to create directory and file with proper permissions . 
Added maxTransfers field to RequestInf class
 Added new constants for unsigned byte ,  short ,  and integer maxima . 
 View :  updated minimum width calculation  (  # 17765 , 7 )  . 
 Added LABEL family to Features class
 Added new method  ' attrib '  with variable arguments for setting shape attributes
Fixed issue where `DefaultStateSerializer` was returning false for default state when `storageId . value (  )  . equals ( storage . id (  )  ) ` was added .   ( Diff revision eac92d2 .  . 23daade ) 
 Increased default size of IndexedRingBuffer from 256 to 128 for Android compatibility . 
 Fixed issue with node indexing in RequestServer . 
 Fixed method signature in PyPackageManagerImpl to match the updated API . 
 Fixed proxy method signature to match the updated class implementation
 Added default value of 200 seconds to maxIdleTime field in HttpConfiguration . 
 Added check for empty GCM token before registering with remote services
 Added new profile drawer items with names ,  emails ,  and icons
Added null literal concatenation in StatementGenerator  (  # 1313 , 10 )  . 
 Updated StorageService . java with token metadata and systemtable updates in server mode . 
 Commit message :  Fixes issue with texture stitching in ForgeHooksClient . java . 
 Fixed bug in RabbitMQEndpoint creation with incorrect parameter types
 Added filter to allow fine grained control over colors in resulting palette . 
 Changed the constructor to private to prevent instantiation from other classes . 
Added method `incrByFloat (  ) ` and updated `incr (  ) ` method in `JedisCluster` to use new method signature . 
 Added import for Google Guava ' s Maps class to improve code readability . 
 Changed behavior of TaskstJob to set user flag to true for all tasks . 
 Added new package name to whitelist :  explore . web . browser
 Fixed bug in ItemUIRegistryImpl . getItem ( String )  :  corrected exception message for ItemNotFoundException . 
 Flushed cache and updated indexing stamp for FileBasedIndex . 
 AddedSuppressWarnings annotation to suppress unchecked warning in MongoDbClient . java
Added optimization for key buffer release in BinaryMemcacheEncoder . 
 Fixes bug by updating comment computation to account for new line numbers . 
 Added support for handling XML attributes in TypedHandler
 Fixed formatting issue in HelpshiftHelper . java
Added new static final string for CHECKIN_SEND_LAST_KMSG_TIME ,  CHECKIN_SEND_APANIC
 Added ability to prevent exceptions when user types  ! foo . 
 Added profiler config to MockTraceContext
 Added support for wildcard subtypes in MediaType . create (  )  . 
 Fixed compatibility issues with Keywhiz CLI and web UI . 
Added persistent UI for vertical and horizontal scroll bars in EditorMarkupModelImpl  ( git commit a62e5aa .  .  . ec5e1f3 )  . 
 Added hashCode (  )  method to DFAState class
 Added method  + 6 to resetView (  )  for soft keyboard opening
 Fixes  # 123456 :  Improves performance of configureTarget method by short - circuiting expensive property and method lookup
 Added new ModuleRootManagerImpl instance with loaded state . 
 Added method to allow subclasses to throw checked exceptions in addition to unchecked exceptions
 Fixed issue with device zones in DenonConnector class
Fixed equality comparison for MethodParameter in SynthesizingMethodParameterTests  (  # 100644 )  . 
 Added support for single point mode with origin specified by options
 Fixed nullability of operands in ReilInstruction constructor
 Added new TesterServletContext to DigestAuthenticator for bug 54521
 Added proper signing key set to PackageKeySetData
 Fixed logging configuration in ProxyServlet to use proper namespace
 Fixed method signature in HttpClientTransportOverHTTP2 to match updated interface . 
 Added support for non - numeric data in PCA
 Commit message :  Added implementation of IdeaTestFixtureFactory interface . 
 Fixed schema version check for plugin schema installation . 
Fix for issue  # 1234 :  updated RemoteBlockInStream to correctly skip to the block position in the checkpoint file
 Update :  Added layout for selected state of WPNumberPicker
 Update to GB unit for memory size in CommonUtils
 Fixed issue with LDAP settings deletion . 
 Fixed bug in WebFilter . java by updating the reload flag for non - sticky sessions . 
Fixed typos and updated cache expiration time in SamlIdPProperties . java from 30 minutes to 1 day . 
 Added option to disable expand events in the REPL . 
 Fixed timestamps in Module class . 
 Rewritten BikeFlagEncoder to handle weighting decision  # 124  ( issue  # 124 ) 
 Fixed bug in ServiceTicketImpl constructor
 Added pattern compile for POS tag in CatalanSynthesizer class
 Changed how the table columns are generated to include ID attribute for each column . 
 Fixed bug causing map to jump to incorrect location when progress bar is clicked
 RenamingMap interface updated to include Serializable interface . 
 Added Touchy movement method to tvInfoMethod in PrivacyService class
 Added OkUrlFactory for more efficient URL connection creation
 Added new logic to determine the longest chunk in the relation triples ,  and updated the longestChunkStart and longestChunk variables accordingly . 
 Added hard reference for non - physical elements and light elements
 Added suffix  . bpmn to BPMN resource names for consistency with BPMN standards
 Added layout inflation for updates list in DashUpdatesFragment
 Fixed constructor issues in WebHistoryItem . java
 Added OnPostPopupListener to ReaderPostAdapter for post selection handling . 
 Added persistId field to Inputs class for persisted inputs . 
 Fixed empty cluster check to improve compatibility with newer Mahout versions . 
 Fixed bug in chunked stream implementation
 Fixed bug in onOptionsItemSelected (  )  to properly handle back button press
 Fixed token text appending in TemplateDataElementType to correctly handle trailing spaces and newlines  (  # 3bb0f99 )  . 
 Added support for IdStrategy annotation
 Added new properties for allDay events :  DTSTART2 ,  DTEND2 ,  and EVENT_TIMEZONE2 .  Also updated the type of the EVENT_TIMEZONE property to TEXT . 
 Changed type of `checkArgument` from `Preconditions . checkArgument` to `checkArgument` to match the new implementation in the updated code . 
 Added ClosureCommand support to RemoteControlHandler
 Changed method signature to fix compile error . 
 Added import for com . thinkaurelius . titan . core . attribute .  * 
 Changed method `getDescription (  ) ` to include the file path in the command . 
 Changed the `searchPaths` variable to include additional views directories . 
 Added placeholder reference in PicassoDrawable for improved performance
 Added assertion for write access allowed in PersistentFSImpl . 
 Added OR IGNORE clause to insert statement for Episodes table
 Fixed default enabled state of PyUnboundLocalVariableInspection to improve inspection accuracy . 
 Changed how the assertions are made for the main (  )  function lines to check for a range of values instead of a fixed number . 
 Optimistic transaction commit :  bypassing local buffer for committed operations . 
 Changed the log file name in DefaultAccessLogReceiver to use the suffix provided instead of the default value . 
 Fixed issue with byte ordering in ByteArrayInput . java
 Added new method to handle case insensitive headers in ApplicationPushBuilder
 Added check for colon in hostname in getServerName (  )  method
 Added PI2 constant with updated value . 
 Added new setting for controlling data and master nodes
 Fixed layout issues in NotificationsListFragment . 
 Fixed a bug in the `loadClassForName` method by changing the `Class . forName` call to use the correct `true` or `false` argument for the second parameter ,  which affects the behavior of the method when dealing with classes that are already loaded . 
 Changed behavior of `ServerSystemTest` to include Compute Engine Database Connection in the expected response . 
 Added backing for Maps . java to use EnumMap instead of Set . 
 Added method to set title text for titled separator
 Added method to load or build resource schema for ResourceModelEncoder
 Fixed issue with suppressed warnings in NoSonarFilterLoader
 Added return value to WindowManagerService # getService (  ) 
 Fixed issue with port collisions in ZooKeeperMasterModel . 
 Added feed icon for queue and all episodes
 Added implicit reference to JspFile tag file
 Added method to registerTaskManager for improved performance
 Added AnimatorSet for RecentsRowFocusAnimationHolder to handle focus loss animation
 Fixed issue with commitAllDocuments (  )  method in LightIdeaTestCase . java
 Added predefined Lexer Rule Properties Dictionary
 Fixed issue with selectByKey method ,  added new method to retrieve all keys . 
 Added continue statement in catch block to handle exceptions in RunManagerImpl . 
 Added expiration time to LocalSession class
 Added scaled image column and updated migratePasswords method . 
 Fixed issue with random datagram packets being received from unintended sources by connecting to the remote address only . 
 Fixed issue with class size calculation in DexFileReader . 
 Added new Comparator for comparing String lengths
 Fixed issue with spell checking in search field
 Added  '  ! exclude '  parameter to BusinessRuleTaskActivityBehavior filter
 Added HTML escaping to NodeCellBuilder for better rendering of node names
 Added method getActualType (  )  to DeferredType
 Upgraded PreStartupStoreUpgrader to use latest GraphDatabaseSettings and DefaultIdGeneratorFactory classes . 
 Changed implementation of `TachyonFileSystemUtils` to periodically poll the master for file status updates ,  increasing the polling interval from 1 second to a generous 1 second . 
 Fixed null pointer exception in create (  )  method . 
 Fixed bug in ReadHandler to correctly handle empty packets . 
 Added support for dropping text and HTML files in the editor
 Changed default password to empty string in DefaultConnectionProvider . java
 Added new child element to XmppConnection class . 
 Added test case for canceling BPW with addon  (  # 1010 ) 
 Added ability to specify boolean parameters for field stats query
 Added DebugComponentOwnershipModule and HMRClient to the JavaScript modules list . 
 Added method  ' rect '  and  ' rect '  with additional parameters ,  and updated ellipseMode method to handle new parameters
 Fixed exception handling for unresolved method annotations
 Added new exception mappings for MBeanUtils . 
 Fixed exception message in ScriptImpl . java to improve error handling . 
 Added propagation of original attributes to WebSocket messages in SimpleHttpProtocol . 
 Added support for endpoint - based JmsExchange creation
 Added com . esotericsoftware . reflectasm . MethodAccess class to module security manager
 Fixed implementation of FizzBuzz method to conform to enterprise - grade standards
 Fixed reentrant lock usage in Daemon class for better concurrency . 
 Fixed method signature for TaskInternalServiceRegistry constructor
 Added ability to publish PMML model updates . 
 Fix for null pointer exception in DnsNameResolverContext . toString (  )  method
 Added import for AstridPreferences for showing list settings popover . 
 Added savedInstanceState parameter to Activity . java for improved state restoration
 Fixed deprecated method call in HeaderTransformer class . 
 Added OkResponseCache import for improved HTTP response caching
 Added import for org . apache . lucene . util . LuceneTestCase
 Added principal name to audit logs
 Changed the `endReached` field from `false` to `false` in the constructor of `TableInputFormat` . 
 Added sort functionality to ConstantPoolTest
 Added default import for Java 14
 Added test for fake failure in StringTools . assureSet (  )  for CI testing . 
 AbstractService :  updated isStarted field to be final
 Added check for anonymous sessions in SessionManager . java
 Added new test class DummyBlockReplacementTest to the suite
 Added property to save splitter proportion in project closure
 Fixed buffer size for CipherWriteBenchmark
 Added new map names and IDs for languages . 
Added double - size constant pool for DoubleConstant class in DoubleConstant . java
 Added getContextPath (  )  method for JspCServletContext
 Changed the encoding size calculation to reflect the length of the primitive type times its count . 
 Fixed bug in ControllerActionTransformer to correctly annotate action methods
Commit message :  Added newerThan option to Yammer Messages Consumer endpoint . 
 Added method `incomingEdgeList ( IndexedWord ) ` to retrieve incoming edges of a word . 
 Added method binding for TargetMethodAnnotationDrivenBinder :  iterate over typeDescription ,  source ,  and target to provide default annotations . 
 Added method toString (  )  for PrimitiveValue class
 Added metaTable (  )  to TreeSync . java for improved scanning efficiency . 
 Added cache for daemon registry services to improve performance
 *  `diff  -  - git a / astrid / src / com / todoroo / astrid / ui / EditTitleControlSet . java b / astrid / src / com / todoroo / astrid / ui / EditTitleControlSet . java` :  Update EditTitleControlSet to use buttonDrawable instead of backgroundResource
 Added support for additional transaction isolation levels in ConnectionPageGeneral . 
 Added method for creating an empty directory and its intermediate directories
 Changed the regular expression for special characters in `WordSplitter . java` to match the updated pattern . 
 Added support for returning complex data types in ReactiveDialog . 
 Added support for null values in OrcStruct . getStructFieldData (  )  and OrcStruct . getStructFieldsDataAsList (  )  . 
 Fixed compatibility issues with latest Android versions . 
 Fixed connection pool allocation issue in JedisSlotBasedConnectionHandler
 Fixed issue with left - click air blocks not being properly ignored . 
 Fixed indentation in UberRDotJava . java . 
 Added method getSuccessHandler (  )  and getFailureHandler (  )  to AbstractAuthenticationProcessingFilter for more flexibility in handling authentication success and failure . 
 Fixed bug in BootstrapTokens . getBootstrapTokens (  )  to correctly handle empty token metadata . 
 Added security check for class loading in ExpressionScriptEngineService
 Added method onStartTrackingTouch (  )  in FillableLoaderPage class . 
 Added error handling for creating account
 Fixed bug in createEmptyFile method to prevent FileAlreadyExistException . 
 Added method return registration id for duplicate event handling in ClientService interface . 
 Added support for overriding the JAVA_IO_TMPDIR environment variable
Added loading of functions from database in OFunctionManagerImpl
 Fixed optimization for match all query for  '  *  :  *  '  and  '  *  '  cases
 Added maximum pane functionality for TextEditingTargetNotebook . 
 Added setAlpha (  )  method to DrawableContainer class
 Added new logic to handle self - hosted sign - in form
 Added implementation for getKey (  )  method in GeneratedMethodBinding class . 
 Added test for invalid surrogate pairs in RealmObjectTests
 Added empty source and char stream pair for CommonToken class
 Added default run time and is ready time to NoopTask ,  updated IsReadyResult to YES . 
 Added exception handling for getBondState (  )  method in BluetoothDevice class . 
 Fixed NOOP method in FilterBase to prevent filter failure when setting unknown configuration attributes . 
 Changed how Windows detection is handled in GraphHandler . java . 
 Added method to set light instead of camera in LightNode class
 Fixed null - terminated string issue in JoglGL . java
 Fixed issue with path indexing in JarFileSystemImpl . 
 Formatting changes :  updated license text from 2000 - 2012 to 2000 - 2015 . 
 Update to use resource id for text color instead of hardcoded value . 
 Added getDisplaySize (  )  method to UIDisplayElement for retrieving display size . 
 Fixed bug in FormAuthModule . java by correcting a null pointer exception in the `getDecodedRequestURI (  ) ` method . 
 Changed the `mScale` field from a `float` to a `double` in `Particle` class . 
 Changed the implementation of `STRING_CAST` from a inner class to a top - level class . 
 Added new method to handle inventory extraction for wood pipe items . 
 Added empty state initializer and copy state initializer methods to PartitionStateGeneratorImpl . 
 Fixes for SSLContext parameters . 
 Added new enum for DateHistogramInterval . 
 Changed how WebSocketProcessorTest . java handles closing of resources . 
 Fixed DNS SRV resolution for XMPP connections . 
 Added check to prevent Chunked encoding in BOSH communication
 Added recent palette menu items for improved usability . 
 Fixed wait time for execution limit test from 1000 to 2000 milliseconds . 
 Fixed handling of XML tags to correctly recognize empty element ends . 
 Added TreeSet for siblings in ResourceBundlePropertiesInsertManagerImpl to improve performance
 Changed the `KubernetesEndpoint` class to use a `KubernetesEndpoint . java` file with a `KubernetesEndpoint` class ,  and updated the ` @ UriEndpoint` annotation to include a `KubernetesEndpoint . java` file with a `KubernetesEndpoint` class . 
 Added empty list of declarations for package  '  “ aPackage ”  '  . 
 Fixed issue with refreshNotes (  )  method not being called after onResume (  )  in NotesAdapter . 
 Added support for L - BFGS solver for multinomial GLM with more predictors than columns . 
 Added new ignore test list entry for  ' fast / dom / DeviceOrientation / basic - operation . html '  to prevent crash . 
 Added retain (  )  method to NettyBuffer . readBytes (  )  to prevent leaks
 Added new JavaCommandLineState implementation for JetRunConfiguration
 Fixed buggy root file marker dirty recursion
 Added android / res / values / strings . xml and android / res / values / styles . xml to the project . 
 Changed 6 lines of code ,  added 1 line of code ,  and removed 2 lines of code . 
 Added PacketHandlerTransport channel for energy - related packets
 Added support for non - streaming ,  client streaming ,  and bidirectional streaming requests in ConcurrencyTest . 
 Rewritten LightPsiFileImpl . java to use view provider for file access
 Changed printing mechanism to prevent System . out from being closed . 
 Fixed bug in ReadPreferenceServerSelector implementation to correctly handle secondary reads . 
 Fixed ratio calculation in DefaultTransform to avoid negative values
 Added new injection mechanism for ClassInjector
 Fixed distribution strategy handling in OutputEmitter . java
 Added Apache License header to AsyncImageView . java
 Removed old Key - Value pair and updated replication status in DKV class . 
 Fixed issue with test failing ,  added  @ Ignore annotation to ignore test . 
 Changed code to fix IllegalAccessException on RoboVM . 
 Fixed layout for QuickStatusBarHeader to properly display expand indicator . 
 Added test for issue  # 615 :  completeWithoutClose
 Added externalizable support for HyperLogLog class
 Added ContainerUnloader interface to DefaultProcessor for improved container management . 
 Added frequently merge policy for FreqTermsEnumTests
 Fixes issue with repair time calculation in OGraphRepair . java
 Fixed warning about random token generation resulting in unbalanced ring . 
 Added new method for executionTimeoutInMilliseconds (  )  in HystrixCommandProperties
 Added WindowManager . LayoutParams . SOFT_INPUT_ADJUST_RESIZE to Dialog ' s getWindow (  )  for better AutoCompleteTextView input
 Added check for session existence before answering registration error . 
 Added import for javax . annotation . CheckReturnValue annotation
 Implemented changes in HttpConnection . java :  added a new method to initiate a connection shutdown with a timeout of 30 seconds ,  and updated the existing method to send a GOAWAY frame instead of closing the connection immediately . 
 Added debug logging for PassExecutorService in DaemonRespondToChangesTest . 
 Added new break char check for BitmapFont . 
 Added empty branch for recent catalog list . 
 Changed `contains` to `contains` in `allSet . toString (  )  . contains ( 1 ns from now )  ; ` to `assertThat ( allSet . toString (  )  )  . contains ( 1 ns from now )  ; `
 Added hasGravity (  )  method to Material enum
 Added if - else statement for handling null  ' tries '  field in MethodDefinition class . 
 Added NOT_FOUND_MARKER to SPVBlockStore ' s notFoundCache for improved performance
 Added project parameter to buildStubTree method
 Changed how the look and feel is set to fix a ClassNotFoundException issue . 
 Added setupStructurePopup method to FileStructureTestBase for improved test functionality
 Fixed issue with missing tile entity data in handled BufferedDescription packets . 
 Added plugin library for world generator and updated security permissions
 Fixed bug in handling of empty zip files in AggregationStrategyWithPreservationTest
 Added OF - 898 fix to replace excessive millisecond values in XMPPDateTimeFormat . 
 Added single SSL socket factory verification across multiple requests
 Changed the number of connections per thread in LoadBalancingProxyClient from 1 to 16 . 
 Added long press callback handling for View
Median of 3 numbers changed from  - 1 to 0
Fixed issue with missing ` @ PreDestroy` annotation in EmbeddedDatabase class . 
 Fixed method signature to make it compatible with newer versions of Java . 
 Changed AcceptEncodingGZIPInterceptor to GZIPEncodingInterceptor and added GZIPDecodingInterceptor . 
 Added support for external browser in ItemActivity
 Added getMaxStackSize (  )  method and updated getSize (  )  method to return maximum stack size in addition to current size . 
 Expanded intermediate adjective conjunct . 
 Added getter for operation counter in MemoryTranslog
 Added check for seeking status before seeking media . 
 Added Im . TYPE_HOME property to VCardEntry class
 Fixed bug in payload processing by updating the mask index to avoid exceeding the maximum value . 
 Fixed bug in longestChunk calculation
 Added import for EntityPlayerMP in FMLCommonHandler . java
 Fixed issue with null value handling in BasicConfiguration # getBoolOrElse (  )  method . 
 Rebased UDP Rebooted method to use `put1 (  ) ` instead of `putUdp (  ) ` for improved performance . 
 Fixed whitespace and quote handling in Rapids . java
 Added logging for AsyncCompletionHandler . 
 Added printing of starting and ending token ,  size ,  and address for each range in the range - to - endpoint mapping . 
 Added new alpha calculation for StackBlurManager
 Fixed null object name issue in OBJLoader
 Added equality override to return actual lock instance for LockHolderImpl
 Added setImageResource ( 0 )  method to MediaGridAdapter for improved image loading
 Increased maximum event length from 2048 to 4096 . 
 Changed behavior of `TestEphemeralFileChannel` to use `getCanonicalFile (  ) ` instead of `new File (  ) ` for `root` . 
 Added ability to focus on resource type combobox
 Added CamelNatsMessageTimestamp and CamelNatsSubscriptionId headers to NatsConsumer . 
Added support for OTR fingerprints in Contact entity
 Fixes issue with non - idempotent commands executed via HTTP GET
 Fixed issue with AmazonSimpleDB client ,  now using AmazonClientException instead of AmazonServiceException . 
Added disposeChildren method to LightStubBuilder to properly dispose children stubs when building a new stub tree  ( commit 474387f )  . 
 Added method signature for HttpTask # createNewRequest in RequestParams
 Added new search results to TraktAddFragment and updated UI for added shows . 
 Added check for reminder reengagement enablement and injection of DependencyInjectionService . 
 Added Javadoc comment for parameter names in DynSqlMapper . 
 Added method to return all descriptors in a package ,  including nested packages
 Added ability to upsert documents in FindAndModify class . 
 Added import for ClassUtils for better type resolution in ObjectIdentityImpl
 Added new action button presentation for Installed PackagesPanel . 
 Added STACKTRACE_MESSAGE constant for improved error messages
 Fixed exception handling in OperatorTakeUntil to handle all exceptions
 Fixed issue with command execution in ListCollectionsOperation . 
 Added method  ' inAnimateZoom (  )  '  to WebTextView to detect and adjust WebTextView during zoom animation . 
 Fixed issue with indexing async method to correctly handle duplicate keys . 
 Update to AbstractVcs :  removed unused FileViewEnvironment getter . 
 Fixed formatting in commit message . 
 Fixed template loader configuration in FreemarkerComponent
 Added support for hiding settings in SettingInjectorService . 
 Added check for null image bytes before converting to Image
 Added concurrent rejection mechanism for AbstractProcessor . 
Added PyClassNameCompletionContributor to complete class names in Python code  (  # 75f94f2 )  . 
 Fixed bug in JingleCandidate parsing
 Fixes handling of touch events in ZoomableDraweeView
 Added configuration option to disable transactions in DBMaker . 
 Added constructor for SwipeAdapterView with AttributeSet and theme resources
 Fixed bug in matchingFragments (  )  to return null when no matches found  (  # 6dee137 )  . 
 Added support for managing accounts in AccountManager . 
 Added provides method for Activity context
 Fixed issue with SETTINGS frame processing in FlowControlTest
 Added flag FLAG_PRIVILEGED to indicate if the application is permitted to hold privileged permissions . 
 Added PreventRareDisastrousClassloadingInFirstCallToLockSupportPark to prevent rare disastrous classloading in first call to LockSupport . park . 
 Fixed empty value check in GroupMapper constructor for improved stability . 
 Fixed issue with JVM decoder not reporting errors for padded sequences
 Added new configuration options for quarry behavior and server database directory visibility . 
 Fixed compile error in ShowsActivity . java
 Changed the key for the write mode parameters from pact . output . writemode to stratosphere . output . writemode . 
 Added ComponentListener for component resized event in PdeBase . java
 Added getLoadBalancerExecutor (  )  method for LoadBalancingRxClient
 Fixed disconnect method signature to match base class
 Added method to handle empty suite notifications and fix empty suite issue in SMTRunnerNotificationsHandler . 
 Added support for empty namespace names in DescriptorUtils . 
 Added new options for FileInStreamIntegrationTest
 Fixed bug in shouldOverrideUrlLoading to properly handle URL loading
 Update brokers info in ProducerZooKeeper :  added new broker   +  newBrokerStringMap  +   ,  updated topic partition map   +  newTopicPartitionMap . 
 RebalancingStore :  added rebalance partition list check for non - metadata store
 Added debug log message for LoadMetadataOptions in BaseFileSystem . java
 Added method  ' onDecorateBitmap ( Canvas )  '  to BaseBitmapTextureAtlasSourceDecorator
 Rewriting labels from old to new format in main . java
 Added new strategy to AllocatorContractTest
  updates BidirectionalTwoOneToManyTest with minor changes
 Restored local changes from stash or shelf and sorted them back to their original change lists . 
 Added more information on how to use ViewPager in conjunction with ActionBar tabs . 
 Added support for icon resource references in ProfileDrawerItem
 Fixed issue with directory listing test . 
 Fixed exception handling in Commands . java
 Added support for widget options in onUpdate (  )  method . 
 Added Pattern . compile (  )  to ByNameArrangementEntryMatcher for faster matching
 Refactoring RefreshableTask to use a single - threaded executor
 Fixed bug in millisecond calculation
 Fixed issue with starting sequence ID in HttpPageBufferClient . java by updating the comparison to use Ints . saturatedCast (  )  to avoid overflow . 
 Added new line of code to handle PREV_APPLICATION_MODE in MapActivityActions . java
 Changed RIL constants to align with new API . 
 Added default value for subnet utils in SecureHttpContext . 
 Changed method signature to improve compatibility with Java 11 . 
 Added default bounds for virtual views in ExploreByTouchHelper . 
 Update MapMarkersHelper to use a new color index calculation for sorted map markers . 
 Fixed bug causing incorrect indexing of certain nodes . 
 Fixed issue with empty refs list in GitRefManagerTest
 Fixed semantical check for function calls on objects . 
 Added import statements for Android app globals and package manager to improve code readability and maintainability . 
 Changed code added an import statement for `org . junit . Ignore` and added an ` @ Ignore` annotation to one of the test methods . 
 Added support for flexible types in ForceResolveUtil . 
 Changed `contains ( FINAL_KEYWORD ) ` to `contains ( FINAL_KEYWORD )  || classDef . getType (  )   =  =  TokenTypes . ENUM_DEF` in `isImmutableFieldDefinedInFinalClass` to improve code quality . 
 Added import for ApplicationInfo in ShadowResolveInfo
 Added method to connectTo ( InetSocketAddress ,  boolean )  and updated getVersionMessage (  )  to include bestHeight . 
 Fixed exception message in GitSimpleHandler to make it more informative . 
 Changed method `getHintFields (  ) ` to return the `List < DBObject > ` directly instead of creating a new `Collections . unmodifiableList (  ) ` instance . 
 Added support for Cykloatlas tiles with 16x7 pixels and 26000 zoom levels . 
 Fixed type annotation issues in JavaCommand class . 
 Added suspend (  )  method to animator for improved performance . 
 Added volatile - write to ManagementCenterService for improved performance
 Fixed bug in rendering method to properly indicate when output model is being prepared . 
Added query test for OrientGraphJdbcQueryTest
 Commit message :  Added a new method to handle the android bug related to getting byte array extra in ExtraHandler . java . 
 Added method to return an empty list when the XML tag is null
 Added error handling for unable to create output directory . 
 Fixed keyword recognition for PythonNamesValidator
 Added support for relative time ranges with integer values . 
 Update to IssueCache :  removed unnecessary cache entry for resource key . 
 Added import for Android Support v7 appcompat library and updated MainActivity layout to use Toolbar and AppCompatActivity . 
 Fixed handling of CMD_START_DRIVER message to transition directly to DriverStarting state instead of waiting for INTERFACE_DISABLED state change . 
 Changed the `TAP_TIMEOUT_MSEC` constant to `TAP_TIMEOUT_MSEC_NEW` and added a new condition to check for taps based on the distance between the current and previous pointer positions . 
 Fixed bug where reader would break on first  - 1 read
 Changed how DNS resolution for google . com is handled in the `HttpClientTest` class to assume that there are multiple addresses returned . 
 Fixed bug in FalsingManager . onBouncerShown (  )  to prevent force session exit . 
 Refactored DynamicAttributeRanking to use refreshMinMax (  )  instead of AbstractRanking . refreshMinMax (  ) 
 Added static method to retrieve WebSocket resource for suspended Atmosphere resources
Added processing exception handling to OrbitRestClient
 Modified Node class to return Token instead of Token getKind (  )  method
Added session scope to MySessionScopedBean
 Added ignore for unsupervised methods for _balance_classes ,  _class_sampling_factors ,  _max_after_balance_size ,  and _max_confusion_matrix_size . 
 Added support for VARCHAR ( MAX_INT )  in MetadataUtil
 Changed how logcat is executed to include the  ' time '  option and added the  '  *  : v '  option to capture all logs . 
Added import statement for org . junit . ComparisonFailure in KotlinDebuggerTestCase for better error handling . 
 Added new resource roots for H2O web and core modules
 Fixed issue where client response was not being properly parsed ,  resulting in incorrect result determination . 
 Added new TldParsingDeploymentProcessor implementation to handle TLDs with multiple tags . 
 Changed the implementation of `determineIndexByTermId` to handle term IDs correctly . 
 Added re - entrancy support for payment gateway processor
 Fixed bug where DragEdge was null in processHandRelease (  )  for surface view . 
 Added enum for TickEvent type
Commit message :  Added import statement for Java NIO file API . 
 Added support for checking for granted authorities in addition to configured attributes . 
 Added isClosed (  )  method to AsyncBatchCursor and implemented it in AsyncQueryBatchCursor . 
 Added hint for background resizing and clipping avoidance in Notification . java
 Added testMultiInsertWithContinue (  )  method in DBCollectionTest . 
 Changed how errors are logged and reported when file fetcher fails due to too many push jobs happening at the same time . 
 Added roster versioning check for XmppConnection class
 Added Set < Intent >  stickyIntents to ShadowApplication for tracking sticky intents . 
 Added OpenCellID marker update functionality . 
 Fixed loading of lib directory in ServerStarter . java
 Added method getTimeout ( Test . class )  to BlockJUnit4ClassRunner for handling test timeouts . 
 Fixed null pointer exception in ConventionAwareHelper . java by updating the null check in the  ' if  ( entry . getValue (  )   =  =  null '  )  '  statement . 
 Fixed compatibility issue with JOGL 1 . 5 . 0 . 
 Fixed naming convention for table updates
 Added method  ' artifactResults '  to class  ' DependencyResolvingClasspath '  . 
 Fixed boolean value for timing tests based on system property . 
 Changed line count from 8 to 11 in `CucumberActivitySteps . java` . 
Fixed issue with root entity serialization in JPQLSerializer  ( commit 131 , 7 )  . 
 Fixed bug where registerForEvent was being called with incorrect parameters . 
 Added import for ConnectivityManager to handle network state in CarrierText
 Added method return value to DeclareVariable class
 Added flash scope interface for maintaining variables between state calls . 
 Added TO - DO comment for PIERRE . 
 Fixed zero - fill for unread data in MuninnPage . 
 Added header  ' Message - ID '  to LocalStore class . 
 Added UNSAFE . loadFence (  )  to prevent older loads being moved ahead of validate in Receiver . java
 Added Azure discovery type and unicast hosts provider
 Added showControllerInFullScreen (  )  method to display media controller in full screen . 
 Commit message :  Fixes issue with recreating log when project level VCS configuration changes . 
 Fix for background data usage allowed check in SyncManager . java ,  improving sync functionality . 
 Added support for constructing RequestParams instances with multiple initial key / value string parameters . 
 Added System . getProperties (  )  to EventTrackerHandler for proper property loading
 Added test for multiple underscores in public suffixes
 FlushAction :  renamed NAME from  ' indices : data / write / flush '  to  ' FlushAction . NAME  +   '  [ s ]  ' 
 Added new static fields and methods for controlling media recording duration and size limit . 
 Update WXEnvironment . java to version 0 . 5 . 2 . 8
 Added new Inspection Tool Registrar implementation . 
 Commit message :  Changed interface `TransactionalTask` to include a new method `execute ( TransactionalTaskContext context ) ` and updated the documentation accordingly . 
 Added MenuDrawer import for improved navigation
 Added configuration for disk threshold decider to improve cluster routing allocation . 
 Added protected access modifier to class members in KdTree . java
 Added check for whole word search in FindManagerImpl
 Added test class for App
Fixed scheduling configuration to use updated task executor class
 Added support for contextual undo view updates in ContextualUndoAdapter . 
 Fixed bug in SSO handling in SsoWebViewClient
 Added import for javax . imageio . ImageIO for improved texture pack reloading performance
 Fixes issue with indeterminate drawable aspect ratio in ProgressBar . java
 Changed class `Platform` to use `com . android . org . conscrypt . OpenSSLSocketImpl` instead of `org . apache . harmony . xnet . provider . jsse . OpenSSLSocketImpl` . 
 Optimized subsampling layer implementation with improved performance
 Added new status offset and increased state buffer length for LogBufferDescriptor
 Added support for writing nodes to disk in the  ' Location '  domain . 
 Added constraint message for incompatible parameter types
 Fixed exception in LiveUsers class . 
 Fixed tab character issue in ClassID . java
 Changed a single line of code in the `TestGremlin` class ,  adding a new test case to send a Gremlin script with a different vertex ID . 
 Added support for handling registration of new Plugin Component with non - null and non - empty Component . 
 Added configurable flag to enable recursive file enumeration
 Improved fitness function implementation for CDFitness class . 
 Fixed formatting issue in Activity . java
 Fixed bug in beta (  )  length check to prevent model rendering in browser if null or too large . 
 Added method to return nfeatures (  )  for autoencoder models
 Added aux field to LocalActor class
 Added shouldShowModuleName (  )  method to PsiDirectoryNode for better module visibility handling
 Fixed numChannels field to 1 for improved performance
 Fixed bug in handling of file inputs by updating InputStreamReader and copyRange (  )  to handle exceptions more robustly . 
 Added method link ( id ,  siteName ,  title ,  summary ,  language ,  infos )  to implement INetworkLink interface . 
 Added test for simple reloading in PolicyMonitorTest
 Added a new static final string  ' LABEL_TEMPLATE_ERROR_MESSAGE '  to handle invalid label template errors . 
 Resolved conflicts in ResolveContext . java ,  added 4 lines of code . 
 Added abstract class PArray to Path interface
 Added implementation for computing languages in FindManagerImpl . 
 Fixed buggy resolveTask . cancel (  )  call to correctly handle cancellation of in - progress tasks . 
 Added check for attribution identifiers and updated code to use them instead of access token for user identification when event usage is limited . 
 Added getExchangesCompleted (  )  method to ManagedPerformanceCounterMBean interface . 
 Added new method  ' contextBindRootScript '  with improved error handling and updated documentation
 Fixed null pointer exception when getting selected template . 
 Changed the description of the BaseOpenInBrowserAction class to include information about the shortcut key for opening URLs of local files . 
 Merged changes to PreDexMergeStepTest . java . 
 Fixed issue with dispatch thread in CompletionProgressIndicator . 
 Added domain lock and magic create domain checks to XmppConnectionService
 Fixed bug in INI template handling . 
 Changed the format of the text displayed for file sizes to use the `FileUtils . byteCountToDisplaySize (  ) ` method instead of hardcoded strings . 
 Added import for Apache Lucene utility class
 Improved implementation of InternalTaskProgressEvent with added method to retrieve task description . 
 Fixed issue with escaping in TextEditingTarget . java by adding else if statement to prevent default behavior when not in Vim mode . 
Updates JBScrollPane to use a new ViewportBorder implementation for improved performance  ( commit 5f4d231 )  . 
 Commit message :  Added support for translating document offsets in PsiFile . 
 Added notifyAll (  )  and join (  )  methods to improve thread synchronization in DrawHandler . 
 Changed icon for silent or vibrate ringer mode in StatusBarPolicy . java from R . drawable . stat_sys_ringer_silent to R . drawable . stat_sys_ringer_vibrate . 
 Added Bouncy Castle provider to ParallelUniverse class for security purposes . 
 Fixed bug in handling of upgrade requests
 Fixes typo in TypeMayBeWeakenedInspection . java
 Added wallpaper crop update for animating windows
 Changed how the backdrop is animated when unlocking directly . 
 Fixed bug in text input method to call listener on rendering thread . 
Added filters property to Emma class
 Fixed bug where delete method would delete folders even if asyncEnabled is false . 
 Fixed import statements for ContentRequestExtractor
 Added method  ' resetProgressiveStart ( int )  '  to reset progressive start state . 
 Added flag FLAG_SHOW_WALLPAPER to PhoneWindow . java for showing wallpaper when window is in focus
 Changed stub loading for PyStubsTest to ensure proper unloading of content . 
 Update OVERSCROLL_DISTANCE to 0 for edge effects improvements . 
 Added BroadcasterCacheInspectorServiceProcessor to handle BroadcasterCacheInspector instances . 
 Fixed issue where onProgressChanged (  )  was being called with values from a previous URL ,  leading to incorrect progress calculations  (  # 387 )  . 
 Added serialVersionUID to ParameterException . 
 Added method to create a file for a given path with write permission on the parent directory . 
 Added visibility check for moved items in FeedItemMenuHandler . 
 Added auto peek height option to BottomSheetBehavior
 Commit message :  Fixes  # 1234567890 by removing unnecessary creation of the walker in `DefaultLenientConfiguration` . 
 Fixed exception handling to include caught exceptions in log output . 
 Added requirement for sub - element XML section to be included in IQ packets
 Added  ' result '  field to ActionAnnotation class
 Added URL field to CompileDriver class for improved compiler logging
 Fixed JFC hack and added missing exports in WarClassloadingDependencyProcessor
 Added lookupKey field for contact display photo URI . 
 Added flag to start indexing tours after downloading some and returning to activity . 
 Fixed syntax error in BuildFileASTTest . java
 Fixed bug in image processing code
 Added new exchange cache implementation for PersistitTransaction . 
 Fixed issue with presence probing for certain user IDs . 
 Added check for started state before calling start (  )  in ValueAnimator . 
 Fixed buggy rendering of custom nodes in PushLog . 
 Added dark icon for delete button in Impassible Roads menu
 Added property values to TestingConnectorSession class
 Changed how `server` and `jdbcDriver` properties are set in `AbstractMetaDataExportMojo` . 
 Fixed indentation calculation for soft wrap after last line
 Fixed table name in GroupTest . java for correct table retrieval
 Added date field to LogEntry class for more detailed log messages
 Commit message :  Added ability to strip HTML tags from description template in DescriptorComposer class . 
Added a new method to get the default block size for large input files in FileSystem and updated the implementation of DistributedFileSystem to use it . 
Commits message :  Added generic overview of ACL handling in graphs . 
 Changed behavior of `HttpURLConnection` to disable caching . 
 Changed method signature from `public static Intent makeIntent ( String className ) ` to `private static Intent makeIntent ( String className ) ` to improve code organization and reduce duplication . 
Fixed compatibility issues in WebService implementation
 Added testReplace method to StringUtilsTest
 Added support for Kafka 0 . 11 . 0 +  consumer API
 Added support for PSK cipher suites in CipherSuiteConverter . 
 Added visibility check for CanvasView ' s content view to improve performance
 Fixed issue with parent role delegation in ORole . java
 Fixed JSON parsing to use YAML Factory for consistency
 Fixed edge iteration support in GraphSONWriterTestSuite
 Added double tap gesture to scroll message list
 Changed how response header values are expected in HeaderAssertionTests . 
 Added method to remove library from project structure
 Fixed issue with project management in SpeedSearchBase . 
 Added option to handle methods in code generation
 Changed handling of proximity sensor negative events to temporarily allow user activity even when keyguard is on . 
 Fixed type safety issue in stubbing :  now using `Sub` instead of `Base` in mocked method . 
 Added  @ NonNull annotation to field  ' partition '  in Operation . java
 Added new HistoryAsTreeProvider implementation for file history . 
 Added method to build input method list locked
 Added support for Spark 1 . 3 in SparkSqlInterpreter
 Added `hasBogusName` field to OpenStreetMapGraphBuilderImpl to handle missing  ' name '  or  ' ref '  tags . 
 Added RadiusMarkerClusterer class for improved BTS pin clustering . 
 Added Android Layout and Strings files to ImportantFiles list
 Fixed method signature for Service . java to match Guava ' s conventions . 
