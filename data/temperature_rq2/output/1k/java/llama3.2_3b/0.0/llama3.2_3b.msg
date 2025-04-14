Added ErrorList.asMap() method to group errors by category
Fixed logger field from final to protected in StatsdLoggerImpl
Improve logging for DependencyResolutionProcess by changing log level from INFO to DEBUG
Added BlockWorkerUtils class with utility methods for tachyon.worker.block package
Fix random number generation in H2TestUtil for reproducibility
Update primary key constraint violation message pattern for MongoDB dialect
Added test for crawling initial URL in CandidateElementExtractorTest
Fix log messages for dropped and received frames in ReceiverUdpChannelTransport
Fixed Util.java method fromStringList to use add() instead of set()
Fix bug where discarding state would fail some tests by invalidating checkpoints on abortion (FLINK-23139)
Update LSMSolver constructor with adaptive Rho calculation and hardcoded value for testing purposes
Fix security vulnerability by clearing and resetting SecurityContextHolder
Added constructor for JedisShardInfo with host, name, port, timeout and weight parameters
Refactor StringStringSyncTest to use a public loadInitialData method
Remove unused OutputSymbol class from NodeRepresentation
Added pager loading and unloading functionality
Fix expected FFDc in DB2TestServlet
Added support for using Transaction Synchronization Registry for rollbackOnly mode
Fix test to correctly propagate non-ignored exceptions
Fix HTTP response handling for JSON content type
Improve SearchDialogAction layout with caption decoration
Optimize capabilities directory implementation by using a concurrentHashMap for storing capability informations
Fix test assertion for CgIsaParamTest.java by removing unnecessary newline characters
Update JUnit assertion imports
Fixed typo in App.java
Added BIAS_FEATURE to BinaryStrFVBuilder
Update management version numbers
Fix nested city object handling in CoreUnmarshaller
Update local hostname retrieval for TachyonClient
Fix closePromise usage in ClientConnectionsShutdown
RefTest updated to extend PresentsTestBase
Fix typo in ArchiveConductor.java
Added verbose mode switch for debug logging
Fix logging level for removed properties in VirtualType
Fix typo in mock() method call
Fix containerless Docker deployable container undeploy logic
Simplify TypeExtractor's visit method for EnclosedExpr nodes by removing unnecessary inner type extraction
Add idle check and terminate extensions before shutting down threads
Added array initialization examples for idempotency testing
Fix manual port exposure for Postgres database container
Update test suite name in XML report generator
Update SimpleDateFormat format in DurationBusinessCalendarTest to match standard date format
Fix variant parsing for alternate field values
Refactor VerifyBase class to remove static variable and use instance variables instead
Fixed SpinnerAdapter import
Fix quotes around sequence names in JDBC databases
Fix synchronization issue with STIXSchema validation
Added support for Byte data type in ColumnValidationReader
Added unique identity to EndpointsLock for leader election example
Fix scrap view handling in Recycler class
Fix unstable test in Chrome with Cyrillic characters
Update resource loading in ApplicationClassLoader to use class loader instead of system class loader
Update XMPPConnection API to remove redundant method calls and improve readability
Update Layer.java to clarify origin-based rotation behavior
Fix limit query behavior when skipCount is enabled
Fix comment formatting in WireCompiler.java
Added started() method to Shell class
Added test for typing a letter and caret position update
Improve JNDIManager by using rebind instead of bind for sessionID binding
Fix server tracer state after sampling
Fixed test fraction value in DeepLearningProstateTest
Refactor ClientFactory to make it public and change default receive mode
Fixed frame rate calculation in AnimationFragment
Fix encoding of attachments in AllureAttachAspects
Fix OpenJDKRule.java to check if tools jar exists instead of checking its file existence
Add 'nobs' metric to excluded methods in ModelMetrics
Improved handling of schema types in GetSchemaInfoCommand
Update version number in CharacterChromosome.java
Fix ControllableRunner tick handling
Fix MojoExecutionException handling and add client close in finally block
Add base directory for unmanaged Tomcat service
Fix TestIgnorance to correctly handle listed services with no version specified
Added string concatenation and variable usage tests for DeepDive06ModelViewController
Fix ObjectUnderFileSystem to correctly handle non-root directories with multiple objects or prefixes
Added support for visiting fragment spreads in VariableTypesMatchRule
Optimize socket client read and write operations by reducing buffer size limits
Fix bug in HibernateDaoServiceImpl where it was incorrectly returning the maximum initial ID instead of the next available one
Enable/disable data caching based on available memory
Enhance RequestAuthenticator to handle GET requests differently
Optimize AnchoredPredictorTest for faster execution by disabling UI and setting local master
Added delay for RealDevice shutdown
Update test file path in JavaFormatterTest.java
Improved documentation for IClassWriter interface, added annotations and properties examples.
Fix formatting issue in CodePrinter.appendOp method
Fix logging statement for when we're not the leader
Added literal method for Percentile series spec
Fix Expect.java to correctly handle failed expectations and provide a clear error message when all expectations fail.
Refactor SiteAwareBuilder to remove unused logger
Update English language rules to include EnglishWordRepeatRule
Update Proto.java to correctly handle boolean values in queries
Fix typo in locale property value
Fix source info usage for Es6RewriteClass method assignments
Disable sound for upload notifications when ring tone is disabled
Fix potential NullPointerException in getChoice() method of AutoCompleteComboBox
Update protocol version in KernelInfoHandler
Added endTag method to XmlEmit class
Added steam running detection and initialization logic
Fix testExceed128Connections to correctly handle exceeding 128 connections
Improve process instance retrieval from KieSession
Warn when multiple kubeconfig files are found, defaulting to the first one
Fix testSMILES() method by removing unused code
Update Conscrypt availability check for Android support
Update query logic in TigerQueryBuilderCanvas
Simplify getter methods in AbstractPatternRule by removing unnecessary accessors
Update BeansXmlDescriptor to use plugin-specific FileDescriptor
Fixed initialization order of Conf class
Update MocoPostRequestAction to remove deprecated InputStreamEntity
Added AttributeExpression class and attrX method for accessing attributes on expressions
Fix test to correctly verify tag names after describe command
Update AbstractTFS class to remove unused method getScheme()
Improve key decryption performance by reducing unnecessary byte array allocations.
Fix test cases for different class loader and close methods
Fix NullPointerException in ExecutorShutdownUtil.shutdownNow()
Add exception handling for getResource() in RepositoryPersistenceTest
Fixed aggregator and collector fragments in ClientInterface.java
Update log level in Log configuration
Added client interface handle and connection ID to MpScheduler
Fix typo in Violation constructor parameter order
Add test for insert and select with null key reference
Added caching and updating parser method to UADetectorServiceFactory
Added tax ID and exporter identification details to CustomsDeclarationTest
Update AccessLevel enum with new minimal access level
Added support for indentation information with default implementation
Disable anonymous access for ResourceServerConfiguration
Fix null pointer exception when scanning files
Update glue packages in CucumberRunner
Fix file creation in TestDataContextUtils.java
Fix: Removed unnecessary print statement in DataSetTableIterator
Added new test classes for multiplexing functionality
Fixed MapRendererImpl opacity setting for RasterLayers
Improve DockerService: handle host not running error when getting IP address
Fix null check in Connectivity.create method
Added null check for connectUri() in ClusterContext.accept method
Fix RedisAppender initialization issue with ServerInfo logging
Improve layout management in ValidateFrame
Improve retry logic for RemoteRequestRetry by handling timeout exceptions and allowing retries beyond initial attempt count.
Fix test for dynamic method invocation with primitive return type
Fix typo in error message for mocking non-private & non-final classes
Refactor ProcessInstanceIdCommand to simplify and improve readability
Fix invalid packet increment logic in RetransmitHandler
Fix typo in index bounds for insertColumn and removeColumn methods
Fix SONARPLUGINS-3097: Handle last empty line in Git blame output
Added Serializable interface to SimpleReadWriteLock
Fix null pointer exception when updating lifecycle manager store
Fixes issue with incorrect loop iteration in CheckConformance class
Update allowableValues method in ResolvedTypes class to use 'LIST' instead of 'string' for enum types
Add serialization check for MojoModel
Add event listeners for join/leave and execute operations in DistributedGroup
Update SSH2HostbasedConnect example to use host-based authentication
Fix history management in SwingTab by removing unnecessary list creation and using existing methods
Update copyright year in ao-dao-base framework
Fixed closePreparedStatement method to properly close prepared statement
Update Index.java to use consistent method name and remove deprecated method
Fix bug in SegmentMetadataQueryRunnerFactory where it was not properly handling results from input.run()
Update ChemObject.java to use HashMap instead of Hashtable for cloning properties
Fix warning in SwingUIFacade's showConfirmDialog method
Fix enum value handling in DGLM class
Fix test failure on integration server by adding ignore annotation
Fix seed attribute handling for suites and methods
Fix exception handling in ResultExporter, reset image name on band element when an error occurs
Fix enum lookup for ThriftCollection by converting input string to lowercase
Fixed nullability annotation on getEmailPath method
Added JUnit assertion method for comparing floats with default tolerance
Fix Path annotation handling in Environment class
Added support for processing CPU and system process metrics in JHipsterMetricsEndpoint
Added delta value for float precision checks in ResizeSettings class
Fix null pointer exception in MainActivity when setting memory cache
Fix invalid ID check in Message class
Fix bug where unpacking was not performed on directories
Improve master process shutdown timeout calculation
Update video mosaic example with new motion estimation parameters
Fix issue #82: validate max request size for large requests
Improve classpath parsing for Windows paths and add recursive deletion support
Fixed PrettyLogViewerPanel to remove date from simtime string
Fix HTTP status code for concurrency limit error
Refactor logging levels in AddonRepositoryImpl for better performance and readability
Added clearViewCached method to allow clearing of cached views
Increase test duration from 20 seconds to 30 seconds
Added flag CACHE_MODE_LOCAL to DistributionManagerImpl for local cache mode
Added box and unbox methods for converting between primitive and object arrays
Improved logging in LoggerPrinter by using Log.getStackTraceString()
Added new rule for 'PRP_RB_NO_VB' grammar error in English language module
Update OperaDriver implementation to reflect vendor support and Selenium integration
Update RadarLevel2Servlet to reflect changes in data type and format
Improve HamcrestCondition's match method by adding a description of the matched item
Remove unused function library loading code
Fix TypeVariableCycleTest to correctly handle type parameter cycles with new annotations
Set cache size for eager expression factory
Fix NaN handling in Sum node
Added cluster configuration entry permissions
Fix null listener check in ApnsConnection constructor
Refactor Finalizer.java to use ConcurrentHashMap for thread-safe tracking of weak references
Fixed sampling method in LinearRegressionTest
Add Transfer-Encoding header for HTTP/2 requests
Update logic for calculating latest watermark in BaseSource class
Fix calculation of next timeout date in TimerTask
Update SourceVersion enum with new versions (1.7 and 1.8) and their corresponding descriptions.
Add tearDown method to reset ContextHolder after each test
Fix double processing of expressions in ExpressionAnalyzer
Fix CheckRange type parameter to extend Comparable<? super C>
Update approvalsRequired property in AsyncMultisubprocessTest
Add Metrics import for HttpClientNoMicrometerTest
Fix db creation in RethinkDBTest
Refactor WebXmlObserver to improve thread context management
Fix null pointer exception when creating new ResultPoint instances
Fix parsing of two-digit year in Japanese calendar format
Added Executor for async execution
Fix serious failure in PolicyFinderModule when evaluating policies
Update assertion percentile in TraversalStrategyPerformanceTest for better performance testing
Fix edge case in BaseHapiTerminologySvcImpl where count is zero or total is zero
Fix issue 122: adjust day calendar for correct end date handling
Fix last available chunk id comparison in GameUpdateTask
Fix racy claim logic and add missing edge cases
Added Vaadin form reference support for AbstractFormPropertyRenderer
Update SdkConfig dependencies
Fix internet connection detection in SdkUtils.java by using application context instead of context.
Fix: skip invalid JPEG marker sequence in RafMetadataReader
Improve span creation by setting as current span
Fix transient field loading in CRUDDocumentPhysicalTest
Fixed method signature consistency in XmlExtractorHelper
Added default constructor and setter methods for ProxyFactoryConfig class
Update Tree class to correctly set result index for each node
Update date format in MultipleRelationsAPITest.java
Fix CLI version mismatch check and clarify non-serializable issue with Arrays.asList
Added logic to retrieve input field value when value is present in ValueComponentHelper
Fix hostname verification in XMPPTCPConnection
Fix ReplaceIfSameOperation to correctly update old value when replacement is successful
Increase default wait time for InteractiveTestJob from 5 seconds to 10 seconds
Add documentation for linear algebraic matrix multiplication method
Fix termId calculation in ArchiveConductor constructor
Fix VerifyMojo to check for orphans correctly
Fix project directory creation in EditorUtilities#copy method
Fixed setCoordinates method to correctly handle coordinate arrays
Fix bug in LogReplication where it was catching all exceptions instead of specific ones and re-throwing a ClusterException with a warning category.
Add alternating background color for DayView in HelloResourcesCalendarView
Fixed datebox modal behavior
Fix bug in BatchLinkingService by allowing null reference resolution
Update ping interval in SimpleRoundRobinLBTest to test shorter cycle
Fix artifact list handling in SingularityMesosTaskBuilder
Fix file system path in RunLocalPayloadTest.java
Refactor logging configuration
Added page story ID field to Photo class for published photos
Add locale check for native library loading in AttachContainerCmdExecTest
Added waiting and timeout functionality for test methods
Update account header selection logic when switching profiles
Fix redirect URL encoding in RackFilter
Fix null checks for registeredService and service in DefaultMultifactorAuthenticationTrustedDeviceBypassEvaluator
Fix logging in TimeScheduler3 to include task name
Update event listeners in FlexyPoolConfiguration class
Fix null pointer exception in getDeclarations method by returning an empty map when declarations is null or empty
Fix pattern test tools logic for inflection and negation checks
Fix potential deadlock in ZipExportDelegate by logging warning with stack trace instead of ignoring exception.
Fix program invocation exception for packaged programs
Fix incubating annotation on StackdriverMeterRegistry
Update Contents class to use root path for navigation
Added request body to JEEContext implementation
Add method to resolve resources relative to a consumer class
Added support for error handling and error writers in RestRouter
Fix typo in IINC instruction generation
Added ElasticLoadBalancerCapabilities health check requirements
Fix RedisModuleMemberRegistrationWithTTLConfiguredIT test to use static method for key generation
Improve performance of DefaultChannelId by reducing string buffer allocation
Added logic for handling ATTRIBUTE_TYPE_GET_OWNERS_EXPLICIT_REQ case in TypeService.java
Fix test server host detection
Update tokenizer to use custom tokenizing characters
Add converters and writer exports for TinyLog implementation
Optimize LZ4BlockInputStream buffer allocation and reading
Added getter for task speed in Listener4SpeedAssistExtend
Optimize blocking wait strategy by removing unnecessary lock and barrier checks
Added hint for full redraw after resize
Fix bug where AsynchronousCloseException was caught instead of ClosedChannelException in NioWorker
Fix race conditions and improve logging for ClassLoaderAdapterCallback
Fix sorting issue in CollectorContextTest.java
Fix whitespace issues in PRTokeniser.java
Added isInputMethodTarget() method to check if text view is input method target
Refactor platform class to use final fields and remove unnecessary instance variables
Update RangeState class to remove transient keyword from callback field
Fix minor typo in BrowserVersionDetectionTest.java
Fix Delegate behavior in MVELLifeCycleManager
Fixed split() method call in Parser.java
Improved locale-awareness in WebPageElement's area finder
Added timeout for waiting on stream closure
Fixed nano formatting in ThreadLog class
Fixed constructor parameter order in Cycles class
Improve server stop logic by using a secure cluster executor
Add JUnit Ignore annotation for EdgeTest.java
Temporary fix for SPARQL LANG handling in ExpressionEvaluator.java
Fix cyclic graph handling in collate endpoint
Update PushApplication entity with new imports and UUID type
Added setter methods for CSV exporter configuration options
Fix FlowableReverse.reverse() to use flatMapIterable instead of flatMap
Remove unused methods from ApplicationClientConfigurationPersister
Added date-related fields and methods for Cookie class
Fix typo in StreamReaders.java, adding support for UNION type
Added method to locate classes in the boot layer
Update DeployMojo to reflect changes in external URL creation behavior
Added initialization flag for beta feature
Fix encoding of client order ID in HitbtcTradeServiceRaw
Added method to retrieve transaction controller for PersistentPriorityProcessorPool
Added WebSocket example with API client usage instructions
Fixed CommerceAddressAdminDisplayContext to correctly handle pagination and search results.
Added ENSURING_CLEAN_SESSION capability for Internet Explorer browser launchers
Fixed logging configuration in LoggingAfterJobListener
Fixed JadeTemplateEngine to use 'this' instead of 'model' in extractAndTransform method
Disable checking for AJAX attributes in repeatUntil method
Fix channel URI in DriverConductor's addSubscriber method
Improve performance of component compaction by reducing unnecessary string conversions
Update import statements for JAXB and XPath
Added event listener registration functionality to ApplicationContext
Fix download id generation in DownloadFromUrlInstaller
Improve reading of compressed chunks in SnappyInputStream
Added new method to initialize typeface from a File object
Fix null pointer exception when calculating molecular charge
Refactor ElementUtils to use a more efficient and flexible way of storing StaxEvent values
Fix typo in AtomicFileWriter documentation
Added fast configuration for QR Code detector optimized for speed
Refactor ClassPathBuilder to only locate required codebases when necessary
Fixed trimTimeAnno method to correctly format time strings
Update FoxPlatformDemoDataDeployer to use PlatformService instead of ProcessEngineService for accessing ProcessArchiveService
Fix null User-Agent header handling in ServiceThemeResolver
Add check for old age and throw BankInterruptedException when execution is interrupted due to it
Added Samsung sensor type for orientation detection
Fix bug where log entries were not being added to work queue correctly
Fix shell error handling to correctly parse boolean value for stack trace flag
Update tsd.query.allow_simultaneous_duplicates setting to true
Fixed keepAliveTimeout item in ThreadPoolEditor
Fix typo in ClusterEventDissector constructor, append 'endpoint=' instead of just a space
Fix null pointer exception when retrieving input type information
Add support for flyWeightTask restrictions and executor availability notifications in QueueListener
Fix bug in SimpleConfigList constructor where it incorrectly accepted a resolved status
Fix timeout issue in IsStillRunningServiceTest with multi-callback
Update connection handling in PresentsClient to prevent closing stale connections
Added runtime configuration logic for LocalTestEnvironment
Fixed hash code calculation in UnsafeBuffer class
Fix Conditional Expectation Estimator Interface usage in RandomVariableDifferentiableAAD
Fix error handling in AbstractSingleImageLayer by throwing a RuntimeException when content type is wrong
Fixed test configuration for BlackScholesDeltaHedgedPortfolioTest
Update WlSeat proxy binding to use WlSeatEventsV3 instead of WlSeatEventsV4
Fix test to correctly handle page fetching and exception handling
Update shard enum constants with new base path
Added getReceivedMessages() method to AbstractReceptionBean class
Enhance loadMetadata method with default LoadMetadataOptions
Improve handling of ClassNotFoundException and LinkageError when loading classes for ServletContainerInitializerRegistrationBean
Fix tuple class meta instantiation method signature
Update version number in ArrayProxy class
Fix null pointer exception in reflexivity checking test
Fix async file write position calculation
Optimize logic transformer by removing duplicate And nodes
Added versioning support for GrailsDomainClassPersistentEntity
Fix batch start awareness in BatchEventProcessor
Fix test failure on conscrypt
Added isMainMethod() check for main method analysis optimization
Increase audit log queue size from 100 to 10,000
Fixed comparison of languages in Main class to also consider external languages
Fix testNonExistentHost to correctly handle invalid hostnames
Fix error handling in SegmentHTTPApi when reading error streams
Fix typo in @see tag
Improve error handling for dropped exceptions and rejected operators
Fix null check for uniqueId in Description class
Fix HALBuilderTest to correctly test HAL response data type
Added support for multiple place types in NearbySearchRequest
Close WebSocket selector and server on shutdown
Fix fields method to ensure fields are passed in pairs
Remove unnecessary access checks when deleting current links from access sets
Fix select option behavior in BrowserTest
Fix server version parsing in ProtocolConnectionImpl
Update NodeCmd.java to include gossip initialization status in output
Fix typo in ShaderSelectFilePropertyHandler, removing unnecessary 'shader/' prefix from shader path
Fix beacon connection URL generation
Fixed calculation of header height in WeekView
Added toString method for PostmarkResponse class
Fix association logic for Http contexts
Fix HetatomImpl to correctly filter atoms by name
Fix non-encrypted connection failure when sslMode is PREFER
Fix issue time parsing in LivecoinAdapters
Update profiler logging to include configuration values
Fix null handling in MapFlatSelectiveStreamReader
Remove deprecated methods from DBCollection class
Update Batch2CoordinatorIT test to use unsorted sorting
Fix typo in exception handling for JerseyServerSentEvents
Fix item re-add logic in AddMenu class
Added synchronization and closed check in LockFreeCache constructor
Fix typo in OperationRequestCompleter, adding address operation separator to completion candidates
Implement peer selection logic with optional override for custom behavior
Fix FollowerState: remove unnecessary reset of last voted for candidate
Update AWS EC2 version in AWSCloud class
Fix watch method to handle exceptions asynchronously
Fix typo in RollbackTest.java author credit
Fix test refresh() method for AllTypes class
Fix KafkaSupervisor exception handling to catch all exceptions
Added missing model to ignore regex in TFGraphTestZooModels
Update TargetType enum values for method return types and parameters
Fix bug where FileInputStream was not being initialized correctly when loading blocks from files
Update JDBC driver version constants for Axibase Time Series Database
Fix: Replace com.google.common.base.Throwables with RuntimeException in FileSystemContext
Refactor PushManager to include all notifications in retry queue and main queue
Added status(), type() and cookie() methods to ResponseWrapper class
Added default property for JPA annotation generation
Fix typo in ZipFileUtil.getZipFile() method call
Added support for PUT requests in ProxyServlet
Improve rollback handling in ObjectifyImpl by catching and logging exceptions
Fix class name extraction in URLPackageScanner
Fix test for concurrent job execution with CountDownLatch
Fix AzureOSImage.java to correctly extract version from provider machine image ID
Added method to process incoming and outgoing wires
Added special case for Type.T_ANY() in NewJavaFileWriter
Update compiler configuration constants
Update RuntimeConstants.java to use major-minor version format in docs URL
Enhance lifecycle API with support for additional closeables in throwing methods
Fix typo in JPerfmonParamsDialog.java, corrected index calculation for 'TAIL' type
Fix minor typo and formatting issues in PdfBoxGraphics2D.java
Fix attachment handling in UnsavedRevision
Fix font size in SpellAndGrammarCheckDialog
Add test for parsing empty string in PathServiceTest.java
Fix Aesh Bug by removing unnecessary variable name splitting
Fix bug in successful authentication test for POST requests
Fix bug in immature class detection by adding missing equals method check
Fix interrupt handling in Emitter thread
Added RxJava support for EmptyInventory
Fix deprecated method in ViewFetcher class
Added Deque-based ImportStack for improved performance and thread-safety
Fix typo in WebSecurityConfigurerAdapter constructor parameter name
Fix CountersReader print functionality
Added createTempFile method for creating a temporary file with delete-on-exit functionality
Simplified DAO methods for EmployeeDAO by removing redundant parameters
Fix typo in AbstractDSLMappingEntry methods
Fix role loading error handling in UsersResource
Increase maximum processing time for JobServiceJmsIntegrationTest
Add 'onload' event handler transformation for FastArchivalUrlReplayParseEventHandler
Reversed subunits stoichiometry sorting for correct formula generation
Improve ShutdownSignalBarrier by adding signalAll() method and removing unnecessary synchronization
Fix command invocation in CommandLineInvoker by removing unnecessary JAVA_OPTS environment variable
Fix round robin execute with failover index calculation
Added JsonCreator annotation for creating GrokPattern instances from JSON data
Fix parsing error when missing type or status values in header metadata
Add support for loopback addresses in ParsedInterfaceCriteria
Fix typo in NodeUtil.java, ensuring correct error message for non-getprop nodes
Updated CustomPathResolver with Interceptor annotation and priority
Update build driver configuration to default internal network usage
Fixed JSON parsing issue in MessageUtils class
Fix typo in exception handling for Exec class
Fix null partition check in KafkaSupervisor
Add Eclipse-specific patching for LiveInjector in Processor class
Update ClusteredObservationBus constructor to allow for default configuration
Fix potential null chain issue in PeerGroup startUp method
Added support for root UFS in WebInterfaceGeneralServlet
Fix event firing issue by updating By tag name to id selector
Refactor IndexedSet to remove unused logger instance
Fix Finder.java to handle error codes more robustly and consistently
Fix assertion in VoltCompiler when loading from InMemoryJarFile
Update class loading logic in Util.java to use current instance's class loader instead of thread context class loader
Fix directory creation in DnsCacheFile constructor
Fixed a bug in GVRViewManager's cullAndRender method by removing unnecessary native calls.
Improved script execution context creation
Added support for DECIMAL data type in JdbcTypeMapper
Cleanup worker state and stop worker
Fix typo in DefaultStompFrame toString method
Added warning for repeated computation of getExpectedLogBelief method
Fix typo in IOException handling
Add marker with title and snippet to map fragment
Update batch synonyms API call to always replace existing synonyms
Added hashCode and equals methods for Resource class
Update stats testing class with immutable lists
Added 'NOT_A_CHAR' constant for Unicode character that should not occur in valid strings
Update GPS directory tag names for horizontal positioning error
Update poller constructor to deprecate direct creation and provide alternative usage
Add test for spaces in LDAP URL
Fix testFilters to correctly handle unsorted cell timestamps
Fix nullValuesOnly logic in GroupProperty class
Improve test concurrency with andThenConsume
Update SpotSceneDirector.java with new log statement and clearing of old cluster object
Fix testTransactionOptimisticCacheMgmt2Db to remove unused code
Update MetricOptions.java with removed Description field
Update low storage space threshold default value
Fix null pointer exception when calling doWrite()
Update default servlet version in Metadata class
Fix test to properly handle multiple IE driver instances
Fixed GrailsDomainBinder to always consider properties as optional
Update gallery configuration locale in CmsGalleryField
Update WordListValidatorTest.java to include öäüÖÄÜßëçèéêáàóòôÈÉÁÀÓÒãñíîş
Add default group ID for FractionList
Update maximum single upload size in SingularityS3UploaderConfiguration
Added flush call in OffHeapValueHolder's writeBack method
Expand UPC-E for purposes of searching
Update BoxedUnit.toString to return an empty JSON object instead of a string literal
Added browser wait logic for Jasmine test runner
Add properties for AutoServiceRegistration configuration
Fix header serialization in MessageSerializer
ST_Explode: Fix geometry column detection logic
Allow LottieAnimationView to invalidate only the changed region of its drawable
Refactor BlockLockManager initialization and add default locks
Fix bug in FileInStream's readCurrentBlockToEnd method
Refactor TemporaryBuildsCleaner to only delete unconnected artifacts
Improve shutdown behavior of FileCacheStore by properly shutting down its executor and calling superclass stop method.
Added service chaining support and worker pool copying in TargetHandler
Fix build timestamp formatting in ExtractorUtils
Fix potential runtime dependency on ServletContext in Sham.java
Filter out non-existent vertices in SparkExecutor and improve payload handling
Fixed siteRole fetch logic in CommerceAccountUserRelLocalServiceImpl
Fix logging configuration for HTTP requests
Refactor AlternatingController to simplify alignment checks
Add typed extern generation check for goog scope aliases
Fix bug where session log records were not being cleared after flushing to file
Added title color setting for RoundedLetterView
Fix BluetoothLeScannerCompat handling of scan results with filters
Fix pattern compilation in Element.java (temporary workaround until release)
Update PerLineTailSplitter class to remove unnecessary public access modifier
Fixed sleep duration in AbstractMindmapsEngineTest
Added security context to PartitionCallableRequest for secure callables
Added error handling for internal failures during authentication
Added methods for creating constructor calls with 'super' and 'this' keywords
Fixed @class attribute resolution in ORecordAttribute
Update TransactionConfidence to reflect new behavior when a transaction becomes newly dead
Fix logging introspector to only print logger and handler information
Refactored DefaultIndexRedirect to remove abstraction
Fix OAuth token handling in REST API
Fixed empty constructor in SomeSteps class
Added onStart and onResume calls in RubotoActivity
Update user creation URL in UserImpl.java
Update ChromeDevToolsTestBase setup method to use 'setUpChrome' instead of 'setUp'
Clear speak service and remove it from dispatcher
Fix typo in method visibility and add null check for defaultBrowser property
Update imports and remove unused Google Common library
Fix pom properties handling in ClasspathKieProject
Fix locking issue with posting change notifications
Fix typo in compareTo method of SingularityTaskMetadata class
Added definitionId field and getter to CaseInformation class
Fix parsing exception handling in DefaultMetricService
Fix size and color initialization in ToggleSwitchBase
Update EsMetricsAccessorTest.java to use host instead of container IP address for ES client configuration
Added notification of preference header selection
Fix minor issue with exception handling in Fmrc class
Fix documentation for CacheCreateConfigOperation to clarify behavior when createAlsoOnOthers is true
Fix bug in TableUtils.clearTable method where DatabaseConnection was not being retrieved correctly
Remove deprecated method and encourage safe exception handling
Fix null pointer exception when setting session in ServletRequestContext
Added fast approximation for atan2 using a formula from Math Stack Exchange
Fix builder pattern usage in HDFSDataset class
Added check for persistent timers being executed from different instances
Convert StringBuffer to StringBuilder in convertToHex method
Reorder stream concatenation in search method for consistency and readability
Fix testExactlyMatchingMethodsArePreferredOverSimilarMethods to correctly handle method ambiguity cases
Fix compiler bug by making close() idempotent for JavacProcessingEnvironment
Update DistributorFlags.java to fix typo in config value name
Add functional interfaces for Throwing, including Predicate, BiConsumer, BiFunction, and BiPredicate.
Fix resource path handling in BaseClassFinderService
Improve UI thread management in ScanProgressDialog
Refactor OpenALSoundPlayer to support multiple loop modes and gain control
Remove unused client bundle source for publications.json
Added copyright notice and license information at the top of PlanConfiguration.java
Improve thread dump formatting for better readability
Fix typo in log statement for getDomainClass method
Refactored cleanup interval calculation for Whirlpool class
Update supported source version range for Java 8 compatibility
Fix bug in LSMSolver where it would incorrectly add diagonal entries when bounds are not set.
Fix authentication retry logic in DefaultClientAdaptorImpl
Update TincanConcurrentQueueHandler to remove unused imports
Update NamedSubResourceSettingBuilder interface method signature
Fix typo in mkdir() call for server temp dir creation
Update ErrorMessage class with Jackson annotations for serialization
Add grace period configuration option to TemporaryJobBuilder
Update TestOperaDriver to use null for controlWindow and currentProduct instead of hardcoded values
Updated debug flag and added timeout constants for Shippo API requests
Added isSupervised method and related logic to DeepLearningModel class
Fix assertion for image dimensions in NewSessionTest.java
Added sudo method for executing consumers under admin privileges
Fixed credential reference attribute definition and added volatile keyword to secret field
Added check for fake bug types and improved error handling in BugInstance constructor
Improve handling of user-defined fields in ControlFile class
Fixed indexing logic for container pages in Solr index
Update test probe builder dependencies and remove unnecessary ones
Fix data header term offset calculation and send logic for heartbeat frames
Added default constructor and new parameterized constructor for Section class
Fix alignment calculation in LogBufferDescriptor
Fix IOExceptions in FileSystem class to include failed URI information
Fix concurrency issue in cache expiration test by increasing clean time
Added RemoteWebDriver dependency to WebElementConverter for caching functionality
Fix missing resource exception handling in ConstantsWithLookupProxy
Improved property mapping in FlatProperties by using LinkedHashMap for consistent ordering
Fix JDBC_PING protocol initialization handling of SQL exceptions
Improved ClasspathUtil#scanClasses method by adding a check for the root package
Update DbTester rule to use AlwaysIncreasingSystem2 for testing purposes
Fix potential buffer overflow when writing byte arrays to ByteArrayOutputStream
Remove deprecated method and update documentation for #isSubscribe()
Update reserved type names in JavaCodeGen class
Fix logging in SoapClient#marshall method
Fix file path inconsistencies in SAMFileMerger class
Fix copyright information in SimpleParameterList.java
Fix test runner thread handling in FlowRunnerTest2
Fix assertion error handling in JSpecRunner
Avoid unnecessary stack traces for ObsoleteVersionException
Updated Config interface constants for case folding behavior
Refactor HuobiAdapters to filter out incomplete orders from limit orders
Update @ToString annotation to include superclass toString in equals algorithm
Fix: Correctly handle file deletion and renaming in OfflineChangeLogHistoryService
Add functionality for repeating commands with Enter key press and navigating history with Up arrow key
Fix InetAddress.getByName() behavior for IPv6 addresses
Fix: Ensure correct handling of single-paragraph blocks by removing parent block after merging children
Fix file not found error in CoverageTest.java
Fixed typo in SessionSummary class, corrected 'performing' to 'committed' for COMMITTING state
Fixed precision loss handling in Timebase resample method
Fix typo in update method call and reorder operations for consistency
Fix GradleCli.java to correctly handle command-line arguments containing the GRADLE_COMMAND environment variable.
Added CPU and memory usage monitoring probes/sensors in MonitoringAgentService
Enable/disable 'Scan all requests' checkbox based on HTTP headers selection
Update ChannelThreadExecutorGroup to use ConcurrentHashMap for executor services
Optimize PathRetractionStrategy to reduce unnecessary checks
Update service name in CreateServicePage.java
Refactor Context interface to remove direct references to Servlet API and deprecate methods
Enhance project converter to handle large file conversions
Fix app name resolution for Android Rhodes Service
Fix leader appointer startup logic for different start actions
Added tryGetDefault method and improved getDefault method documentation
Fixed JDateTime hashCode() method to include time component
Update pattern compilation for nullable and non-nullable annotations
Simplify CollectionSubject.create method signature
Improved string comparison in ObjectUtils by using toString() instead of toStringOrNull()
Update auth mechanism mapping for DIGEST and BASIC authentication
Fix typo in setUp method of TestWithStun class
Add capability for proxy configuration in NewSessionPayload
Fix bug in DefaultLogicResult by propagating exceptions properly
Add test URL suffix for colon issue #4
Add test for assertArrayType method with null input
Added role check for view rename test
Fix Ctrl+C handling in CliTableauResultView
Added isRuntime method to ThreadPoolResourceDefinition enum
Update Stopwatch class to use custom Voltpatches implementation
Fix parsing errors in schema equivalence checker
Update AndroidTestFinder.java to include instrumentation test classes to be run by InstrumentationTestRunner
Initialize ResourceAccessor in BaseLiquibaseTask
Simplify ResourceConfigApplier interface method signature
Update Loader.java to use standard Java NIO charset constants
Fix: Ensure that an AssertionError is always thrown when verifying a reference in Postconditions class
Added attributes field to BodyCells model
Enhance BakeWatcher to watch for content, template and asset changes
Update logging level in test case
Fix extraction of DSOs in UnpackingSoSource by ensuring directory is writable before attempting to extract
Fix image connection liveness timeout check and update active transport count
Fix bug where leaving a room could result in incorrect occupant information
Fix item removal logic in FastAdapter
Fix PinListSync logging to warn instead of error on sync failure
Update S3_PING protocol to use HTTPS instead of HTTP for signature generation
Update copyright year in RawType.java
Fix onClose method to return a Mono instead of itself
Improve concurrent access for persist requests and jobs in DefaultFileSystemMaster
Fix LRA filter component initialization handling of potential RuntimeExceptions
Fix hidden node instance logic in NodeInstanceImpl
Update CommandLineParser.java with improved documentation for ngram rule and added link to wiki page
Fix potential 3-byte surrogate / 6-byte surrogate pair handling in StringSerializer
Fix illegal state exception handling in DatastoreImpl
Fix CountCommandResult getCount method to correctly handle numeric values
Deprecate compilation from project files
Fix permission check for locking resources in CmsLocaleGroupService
Update AbstractPredefinedQuery class with removed Nashorn API dependency
Fix NullPointerException in JRubyAdapter's printStackTrace method
Fix potential Elasticsearch issue with large dataset flushing
Added Elasticsearch container environment variable for disk threshold configuration
Fix null pointer exception in ByteSelectiveStreamReader by setting all fields to null when closing
Fix null class name logging in App.java
Update PropertiesEscapeUtil to remove unused imports and simplify escape logic
Fix test to wait for all nodes to reach safe state before invoking partition operation
Fixed nonce calculation in BTCEBasePollingService to ensure synchronization across processes with the same API key
Refactor ParserTransformer to use a custom ClassWriter with parent class loader
Refactor shutdown method to handle repeat requests and improve logging
Fix connection close logic in LockJDBCConnection
Fix null pointer exception in SurfaceLayerGL by making surface a final field
Fixed unnecessary SimpleDateFormat usage in FramerManager.tick method
Fix null check for response.getAllHeaders()
Fix ASN.1 decoding of ECKey sequence
Simplify HTTP Security Preparer role handling
Improve property evaluation in Interpreter.java by short-circuiting early termination
Fix file system initialization for default ClassPathResourceLoader
Improve PluginManagerInterceptor's configureComponent method by allowing preExecute listener to modify mojo configuration
Fix invalid URL check in VerifierServletTransaction
Trim whitespace from StoryState output stream
Fix MockTest.java to correctly initialize int1 field
Refactor BranchTemplateServiceImpl to simplify and improve performance
Fix closing of writer in Delombok class
Fix @ClassRule repeat test handling to prevent unexpected behavior after repeated tests
Update PlaceRegistry.java with OID changes and logging improvements
Improve Period calculation for fixed month ranges
Fix null check for optimize in RegisteredLookupExtractionFn
Update ServerRemoveHandler to require runtime for server removal operations
Fixes issue with varying lengths of share values in Shamir Secret Sharing implementation
Added utility methods for converting camelCase strings to human-readable names
Added LTHeadMenu and ContextMenuInterceptor for LanguageToolMenus
Fix null check for chains in KeyChainGroup constructor
Fix private constructor to allow for additional script hash creation methods
Update JUnitTestRunner to use NSException instead of JavaLangThrowable
Added AuthoritiesConstants import for role-based access control
Fix null pointer exception in InstructionSet.isDeclaration()}
Added toString method for BootChargingStationCommand class
Refactor JsfHelper.java to improve type safety and readability
Refactor HelpFlags to use Java 8 features and simplify configuration iteration
Added ToolStrip property to FeatureAttributeWindow for toolstrip at top
Add pre-crawling plugin execution and fix index state calculation
Fix mkdirs logic in OutputFileMixin to handle existing directories correctly
Added new methods for getting first header and value in BasicHttpClient
Update DirectLogFetcherTest.java to use writePkg() method instead of write()
Added delay in system exit to provide more informative exit message
Fix null pointer exception when setting background color in CollapsingTitleBar
Fix memory registration failure handling in NatRegMrCall
Update Activity.java to match new API schema for 'type' property
Refactor CampAST.emit() method to return a final String instead of StringWriter and PrintWriter
Add test for replicated table optimization rejection due to coordinator NLJ node being a child of ORDER BY node
Added sequence number field to CommonLogFields interface
Fix URL scheme validation in BoxBrowser class
Refactor CanvasTestBase to make it abstract
Update NotificationConsumer interface with no changes
Improved locale-awareness in DatabaseValidator by using Locale.US for field name conversion
Improve disposal handling in DisposableHelper by making it thread-safe
Fix typo in Streams class methods for input/output streams
Added JPA bootstrap logic to Setup class
Update observable creation with execution hook
Fix shuffle logic in DRF to prevent non-OOB rows from affecting variable importance
Fix row count mismatch issue when executing FindAllQuery
Enhance EeClassLoader's findClass method to simplify its implementation and improve readability
Add verbose logging for icon enrichment process
Increase send and receive buffers for unicast socket in LargeState test
Update AndroidSdk.java to use available platforms instead of fixed version
Fixed stale element reference exception response code
Refactor ModelUtils to use a more flexible and reusable method for checking if an object is an OpenEngSBModel
Fix expiration date calculation for cookies with max-age greater than 1 second
Fixed private constructor in JAXBUtil class
Refactor BrowserMobProxyTest to filter out unnecessary requests
Fixed invalid layout issue in C3DTextFieldSkin
Fix test annotations for context classes
Update TransactionRuleBuilder to reflect its new purpose as a transaction rule builder
Improved configuration loading for ActiveJDBC
Update DiameterStackMultiplexerAS7 interface with @Local annotation for EJB 3 compatibility
Fix null token issue in AmazonServiceCatalogService
Added phenotype validation and individual creation retries to Engine configuration options
Update REST API to use correct Media Type for multipart/mixed content
Update KafkaClient interface to refresh configuration on server and group ID modifications
Improved LazyPatternBy class by adding toString method and ensuring thread safety through synchronized methods
Update runtime property map for app logo and href logo
Fixed version calculation in ClientSession
Fixed random ID generation in SearchResultServlet
Fix testAnimation() method to mark it as ignored
Fixed ListItems implementation with correct type parameter and added properties method
Refactor MutableBySettingFieldOfField class to use IndirectField objects instead of direct field assignments
Update imports for Jakarta Servlet API
Added HGVS code generation for SNVs in Variant.java
Update m2.repo property in ArchetypeTest.java
Fix PathMatchingSimpleStorageResourcePatternResolver to always add matching resources to the result set
Added snapshot data and flag for receiving first fragment task message in JoinProducer
Update ModelObjectWithChildren interface with new since version number
Fix gap activation logic when delay generation fails
Fix signed/unsigned mismatch in EntityBuilder channel retrieval
Fix null pointer exception when adding table references to change log graph
Fix invalid Unicode character handling in DatabaseUtils.java
Added isInEditMode() check in init method of RippleView
Refactor Helper class to remove redundant method and improve code quality
Added tree processing callback for TSMeta
Update DeploymentBuilder constructor to use RepositoryService instead of RuntimeService
Add test for exact two queries with no rows and other thread execution
Fix bug in SqlgStartStepBarrier where traverser generation was not being called correctly
Fix List constructor to correctly handle list tags
Fix handling of exceptions in Puller's message processing
Fix typo in log message for low storage warning
Fix ElementSelectingTest: Correctly toggle radio button selection
Update GradleCompatibilityExtension with new Gradle version in GRADLE_VERSIONS list
Added SPARQL service description namespace and endpoint property to RdfLexicon
Fix memory calculation in Utils class
Improved batched statement support for LoadDataChange
Add support for transient modifiers in AdapterUtil
Fix typo in parsing MetricParams from string
Improve UsingToStringOrdering to use natural order of string representation
Improve global AST transformation configuration parsing
Refactored testMultipleGraphs method to use a smaller thread pool size
Fix remote link close event handling
Fix timeout issue in TestHiveRecoverableGroupedExecution#testCreateBucketedTable
Fixed global lock for bitmap decoding to prevent excessive memory thrashing and potential OOMs.
Fix predicate creation method for 'not' function
Fix: Ensure transitions are cloned with correct duration
Fix null column handling in Index.toString() method
Fix event handler logging issue by adding event body
Fix XML parser issues in RSS test cases
Back off drain() call in SoundManager to prevent thread lockup
Fix javac6 compatibility issue and add more informative error messages for LombokFileObjects
Fix year/day factory tests to include equal sign in loop condition
Update UserProfile constructor with deprecated alternative
Simplify IO object read in EngineTest.java
Added @Ignore annotation to StageTest class and imported org.junit.Ignore
Fix parallel test category for WanRESTTest
Update ExclusiveBufferClaim class to use byte instead of short for flags field
Fix typo in OptimalCECPParameters#getUserConfigTypes method
Fixed rendering mode handling in ScheduleRenderer
Synchronize access to thread-local Looper instance in ShadowLooper.myLooper()
Update AbstractFile.toString to include directory separator
Fix null pointer exception in CmsClientReportWidget.append method
Fix flagEncoderMap uniqueness check and add error handling for missing profiles
Convert separator in file separators to forward slash for consistent GitHub site plugin behavior
Fix deserialization of response in ClientDelegatingFuture
Added imports for Condition and SelenideElement classes
Added data setter and getter for SessionInfo class
Update CPU and scheduled time calculations in PlanPrinter
Added private constructor to prevent instantiation of CloudSpannerOAuthUtil
Update MySQL writer commands SQL formats for table creation and truncation
Added support for URI creation from file paths in FileSystemSupport interface
Use weak hash map in SlaveReconnectionWork to avoid leaking Slave objects
Fixed random number generation in email output
Refactor WeaselProgram to move to top-level package
Fix missingResource substring extraction in ModelTest.java
Optimize BCrypt comparison for constant-time equality
Fixed node mapping logic in DocumentGraphMapper
Fix unused variable and simplify JCommander usage
Fix default method handle invocation with proxy object cast
Added underscored ISO code mapping for I18n class
Fix type parameter check in Check.java
Fix bug #630111 by reading the full class file header in JDK7
Fix JoinRequest packet version mismatch logging
Fix DagManager: correctly check for running jobs in a DAG
Fix shutdown hook registration in NodeService
Fix formatting of LinkedIn identity URL in PsLinkedin class
Update default worker count in WorkerOptions class
Update FormatterMojo.java to include exception skipping in formatter XML configuration
Added `upgrade()` method to lift security restriction on underlying channel
Added transaction support for Bidirectional and OutBound resource adapters
Fix recipient provider logic for AbstractBuild and RunResult cases
Fix audio connection handling in WebSocketClient
Fix data deserialization in InstallRequest
Fix null check in JoinMatcher.accept method
Fixed source code extraction logic to correctly handle line numbers.
Fix Windows compatibility issue with ChronicleMapBuilder
Refactor HiveMetadata to filter partitions based on constraint predicate
Update LakeBTC exchange specification with corrected SSL URI and host
Added connection timeout and read timeout configuration to DemoApplication class
Improve RefreshListener behavior when no ServiceMatcher is provided
Suppress exceptions in ScheduledReporter's report method
Fix calculation of end column in JLanguageTool
Fixed PairBasedRestfulRouter's path determination logic to correctly identify ID elements.
Fix private field naming convention in GoogleCodingConvention
Update WebSocketSubscriptionTransport constructor with new factory parameter
Fix ViewPropertyDescriptor.newInstance() method to always return null
Fix typo in intent extras for file list activity
Fix null adapter in AbstractPreferenceFragment.onCreateAdapter()
Fix sleep interruption logging level from error to debug
Added Jackson annotations for HostIdentifier constructor and getter methods
Fix CSVWriter.java to correctly handle extra escape characters
Update BasePairParameters class to remove unnecessary field and add new one
Fix parsing of 'time' field in AbstractResultParser
Refactor normalise() method in Type.java for performance and readability improvements
Added default artifact type configuration for SvnConnectorConfiguration
Refactor CommandManager to use TreeSet instead of HashSet for command names
Fix TypeDescription#findVariable() to return UNDEFINED when no enclosing source is available
Added CacheResultUnavailable exception for cache result unavailability
Added Facebook Messenger extension support and fallback URL fields to WebButton class
Enhance SessionDataManager to include parent node in readItem call
Improved logging in DiscoveryEntryStoreInMemory for better debugging and tracing capabilities
Improved performance in AbstractRepository by using Spring Specifications Executor for counting entities
Fix storage clearing in StandardTitanGraph
Fix channel URI construction in AeronCluster
Fix null check for index name in CFMetaData class
Update testCovtypeGBM method with new annotation and file path
Added name field and getName/setName methods to GeoJsonObject class
Added conductor cycle threshold configuration option
Fix connection cleanup issue when execution service is not live
Fix typo in domain configuration file existence check
Clear task fork list on task start
Fix compiler bug by making close() idempotent for JavacProcessingEnvironment
Fix GitHubProvider to handle missing display name
Added readForRecycle method to ByteBuf class for efficient reuse of buffer data
Fix null pointer check for 'comment' field in MediawikiApi implementation
Fix error propagation when error is not on subscribe
Update copyright year in RepackageMojo.java
Fix closing of DBCursor after iterating over results in GridFS.java
Add annotation support for Documented
Added constructor for Meta class with no assigned ID
Fixed bug in ObjFunction's sigmoid calculation and division by sum
Update SSL/TLS protocol version in SslTest.java
Update GermanSpellerRule with corrected spelling repls
Fixed DataIO.unpackInt method to use correct input stream parameter
Update DevelopmentPlugin to use a single VelocityEngine instance
Improve thread usage in PlannerBenchmarkConfig for 'AUTO' mode
Added managed attributes for slow and very slow invocation thresholds in ServiceStatistics class
Fix OpsAgent registration and stats agent startup order
Cleaned up path generation in Fingerprinter class
Added null check for manual provider in Man.java
Improve zip file handling in Sync plugin by adding support for extracting zip files and copying non-zip files instead of unzipping them directly.
Fix test assertions for UserExtensionsTest class
Added email and subject to Intent for sending mail via chooser
Refactor HibernateBundle constructor to accept entity and additional entities separately
Add support for setting ACL on multipart uploads
Fix scroll event handling in MapController
Fix XbaseTypeComputer initialization logic
Fixes issues with modified operands and index operand handling in VerificationTransformer
Added missing registerPlugins call and added debug version display
Add version number to transaction output
Fix showMenuItems method to correctly call JMenuOperator and JMenuItemOperator
Fix null pointer exception in ExceptionProxy#constructExceptionForClass method
Fix typo in ObjectiveCImplementationGenerator, correctly initializing forwardDecls set
Warm up partitions in small cluster test
Update Scheduling.java to make singleton option always true
Update dependency changes calculation in AbstractBuild
Update exam reactor strategy
Fix scheduling latency mark timer with fixed delay instead of rate
Optimize logging for SelectorOptimizer
Fix test exception handling in TestUtil.java
Added alt text discriminator for Selenium image rendering in DropDownImageListEditor
Update cluster node disk usage stats query
Fix type argument resolution in AbstractBeanDefinitionBuilder
Update VagrantState enum to include provisioning command for running state
Added ReadOnlyContext emission of Notification
Improve AvroUtils.java by adding a check for record fields size mismatch when comparing schemas
Improve error handling in AbstractRemoteCommandObject
Fix empty string check in DocbookBuilder.java
Fix server closing issue in PythonPlanStreamer
Remove unused ClassDescriptor fields and methods
Add toString() method for FluxMaterialize
Added exception handling for config loading in AutoReloadingConfiguration.init()}
Fix bug in AbstractMappingStrategy constructor where logging exception was not properly formatted.
Update author name in ProductRegisterEndPoint.java
Fix: Added default constructor for PUBREC class
Add test for large date format handling in PactDslJsonBody
Fix: Removed unnecessary print statement in CollapsePropertiesTest.java
Update environment detection logic for standalone mode and profile checks
Fix inconsistency in retrieve behavior for ProfileType and update documentation to reflect change
Update security feature set check in GrpcServerComponent
Update TimeZones class to include IBM's GMT timezone ID
Refactored NUM_FILES_PER_CHUNK to improve load leveling and reduce overhead
Update language translation service to include optional monolingual and parallel corpus parameters in request body.
Update minimum server version check in NotifyTest.java
Fix expired listener removal in PushManager
Add logging statement for FeatureRequirement assumptions
Fix StrictMode configuration in MainActivity.java
Fix JSON deserialization issues in SnapshotRestore class
Added toString method and default value for location field in ArtifactQuery class
Update DiskLruCache documentation to clarify entry creation and editing behavior
Fix thread count in AsyncScalr service initialization
Update FxmlLoadResult class to accept Node instead of its subclass in afterNodeLoaded method
Improve CaddScoreParser by handling last chunks of raw and scaled values correctly
Added new method to InternalKieBuilder interface for building all KieBases with optional class filter
Clear attachments from deployment units after stopping sub-deployment
Fix removeAll method to correctly remove elements from JsArrayList
Fix test setup for JsonSchemaValidationTest to accurately reflect repository behavior
Fix null check for lazily loaded constructor
Update candidate term ID calculation in Election class
Add conditional for noSummary property in AnalysisRun class
Update OrganizationDto class with removed unused field and minor formatting adjustments
Fix idle wait logic for OperaDriver when navigating between URLs
Fixed SwissKnife toDate method to use LocalDateTime instead of manual parsing
Fix null pointer exception in Script.java's execute method
Fix bug in PartiallyConstructedObjectAccess detector where it incorrectly reported bugs for private methods.
Update copyright year in AnalyzerPhase.java
Fix corrupted hint name handling in HintedHandOffManager
Fix IFFImageWriter progress calculation
Added path to resource model
Update OPEN_FILES_LIMIT configuration value
Added backpressure handling for non-writable channels and added delivery of pending messages when channel is writable
Fix sharding properties validation for non-integer values
Update AndPermission.java to accept Context instead of Fragment
Fix test assertion for acquired locks in PidRuntimeFactoryTest
Fix rename operation in MasterInfo class
Added cache for access checks in Permit class
Fix potential buffer overflow in PBDSegment's truncated entry writing
Fix cancellation logic in SqlStageExecution
Fix bug in ThriftClusterImpl where it was printing the CfDef object instead of using it
Fix bug where operator would not close when all clients were complete or the exchange client was closed.
Updated TreeNode interface to return a boolean indicating whether long click was handled
Fix bug in ConfigException where ValidationFailed could throw an empty exception
Add temp config file deletion on exit for exposure server test
Update SubResourceSetting.java with removed imports
Added support for PriorityResource in SingularityResourceModule
Fix null pointer exception in FunctionContext class
Fix XMLRuleHandler to correctly handle end position corrections and prevent SAXExceptions when marking tokens outside of patterns.
Update OneShotBenchmark configuration for improved performance
Enhance AddPrimaryKeyGenerator to support DB2 databases
Fix publication logic to correctly advance term IDs and update log metadata
Add Data annotation with new static constructor option
Add validation for taking snapshots when in JOINING mode
Added conditional URL addition for tools.jar in JspcMojoClassLoader
Fix debug logging in BitcoinSerializer to only enable when debug level is set
Added compareAndSet method for atomic sequence updates
Fix assertion for inserted student in ModelTest.java
Reduce expected refresh count in PeriodicalRefreshStrategyTest
Refactor AsyncExecute to properly set and use execute environment
Update Search UI with new functionality: update user information after query evaluation
Fixed DefiniteAssignment.java to correctly calculate initial store size
Update KieControllerSwaggerIntegrationTest to account for multiple possible controller documentation titles
Fix safe mode calculation and update logic
Add test for auto-compaction with special characters in PDF417 encoding
Improve validation for timeslot duration in SseBuilder
Increase timeouts for NettyTransportTest tests
Fix empty access token validation in MPRestClient
Fixed attribute filtering in CmdLineAnnotatorUtils to correctly filter output attribute names
Fix task manager logic for partition writers availability
Simplify cluster formation in MultipleCacheManagersTest
Fix BeanSpec handling of generic types with no type parameters
Fix testGetPerformanceNumbers to correctly handle not found cases and empty containers
Refactor layer state management for efficiency and clarity
Fix TODOs in ResolveExpressionTypesPass for better error handling and performance
Added new fields to PolicyEvent class for policy rule URL
Fix LogResource.java: added X-Text-Delivered header
Added onRequestPermissionsResult method implementation
Add native implementation for Thread.interrupted() method
Improve error handling in RollingUpdateOpFactory for port conflicts
Update max text length response in ApiV2.java to use anonymous config value
Deprecate ContentMap methods for adding support of external map operations
Fix newline and tab characters in URIs
Fix whitespace trimming in XWikiXHTMLWhitespaceXMLFilter
