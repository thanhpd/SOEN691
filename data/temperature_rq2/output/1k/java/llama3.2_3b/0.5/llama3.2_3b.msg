Improved ErrorList's asMap() method for better performance and categorization of validation errors.
Refactor StatsdLoggerImpl to make logger field protected instead of private
Automatically resolve transitive class dependencies with improved logging
Added BlockWorkerUtils class with utility methods for getting master and worker addresses from Tachyon configuration
Fix random number generation in H2TestUtil for reproducibility
Update primary key constraint violation message pattern for MongoDB dialect
Added test for crawling initial URL in CandidateElementExtractorTest
Fix logging of dropped frames in ReceiverUdpChannelTransport
Fixed Util.java method fromStringList to use add instead of set, as it is more efficient and correct behavior.
Fix bug where discarding state would fail some tests (FLINK-23139)
Fixed LSMSolver constructor and added adaptive Rho calculation
Fix security context management in SpringSecurityWebProcessingContextBuilder
Added constructor for JedisShardInfo with host, name, port, timeout and weight parameters
Refactor StringStringSyncTest to use public method for loadInitialData
Remove unused OutputSymbol class from NodeRepresentation
Added pager loading and unloading functionality
Fix expected exception for DB2 connection downgrade
Added support for using Transaction Synchronization Registry for rollback-only mode
Fix test to correctly propagate non-ignored exceptions when overriding default ignore exceptions
Fix HTTP response handling for JSON content type
Improved search dialog layout with caption decoration
Optimize capabilities directory implementation by using a ConcurrentHashMap for storing capability informations
Fix test assertion for CgIsaParamTest by removing unnecessary newline characters.
Update JUnit assertion imports to use MatcherAssert for better compatibility and readability
Fixed indentation and added semicolon for final keyword
Add BIAS_FEATURE to feature set
Update management version numbers
Fix nested city object skipping in unmarshalCityModel method
Update local hostname retrieval in TachyonClient
Close promises cleanup and optimization (TODO: remove unnecessary promise)
RefTest now extends PresentsTestBase instead of PresentsTest
Fix typo in ArchiveConductor.java
Added verbose mode switch for logging debug info to System.out when verbosity is true
Fix logging level for removed properties in VirtualType
Fix typo in mock() method call
Fix containerless Docker deployable container deployment logic
Simplify TypeExtractor by removing unnecessary lambdas
Implement idle check and terminate extensions before shutting down threads.
Added array initialization examples for idempotency testing
Fix port exposure for Postgres database container due to regression in 1.14.0
Fix typo in test suite name
Update SimpleDateFormat format in DurationBusinessCalendarTest.java to use MM dd instead of MMM dd
Fix variant parsing for alternate field with hyphen suffix
Update VerifyBase class with improved user variable generation
Fixed SpinnerAdapter import
Fix quotes around sequence names in JDBC databases (Liquibase-XXX)
Fix synchronization issue with STIXSchema validation to avoid unexpected SAXExceptions
Added Short and Byte data types to ColumnValidationReader's supported parameter types
Add support for dynamic namespace and unique identity in LeaderElectionExample.java
Fix scrap view adding logic to correctly handle viewType count changes
Fix unstable test for ConfirmTest with Chrome browser
Fix resource loading in ApplicationClassLoader
Add sync and async packet listeners to XMPPConnection interface
Update Layer interface to reflect origin around which rotation is done
Fixed MongoDBCollection query result calculation for pagination.
Fix comment formatting in WireCompiler.java
Added started() method to Shell class
Added test for typing a letter and its position in inline CSS text area
Fix binding method for JNDIManager to use rebind instead of bind
Fix server tracer state in Zipkin SpanListener
Update test case with new fractional value for DeepLearningProstateTest
Renamed ClientFactory to public class and updated constant value
Fixed calculation for target frames to match FPS frame rate
Fix encoding of attachments in AllureAttachAspects
Fix OpenJDKRule.java to check if tools jar exists before checking for file existence
Add 'nobs' metric to excluded methods in ModelMetrics.java
Improved handling of schema types in GetSchemaInfoCommand
Update version number in CharacterChromosome class
Fix ControllableRunner's tick handling and reset behavior
Add finally block to ensure client is closed after exception handling
Add base directory for unmanaged Tomcat service
Fix TestIgnorance to correctly handle services with no version specified
Refactor DeepDive06ModelViewController to demonstrate string concatenation and MVC principles.
Improve ObjectUnderFileSystem: handle non-root directories and improve condition checks for creating metadata.
Added support for visiting fragment spreads in VariableTypesMatchRule
Improve socket client read/write handling for better performance
Fixed HibernateDaoServiceImpl to correctly retrieve maximum app ID
Optimize DeepWater model memory usage and enable caching when possible.
Fixed RequestAuthenticator to handle GET requests correctly
Optimize AnchoredPredictorTest configuration for local testing
Added delay for RealDevice shutdown
Update test file format in JavaFormatterTest.java
Fixed typo in IClassWriter interface documentation
Fix formatting issue in CodePrinter.appendOp method
Fix logging statement for when we're not the master
Added literal method for Percentile series spec
Fix Expect.java to correctly handle failed expectations and add TODO note for TestRule override
Fixed logger annotation in SiteAwareBuilder.java
Update English language rules with new rule: EnglishWordRepeatBeginningRule
Update Proto.java to correctly handle boolean values
Fix typo in locale property value
Fix source info usage for Es6RewriteClass method assignments
Disable sound for low-priority upload notifications
Fixed exception handling in AutoCompleteComboBox.getChoice() method
Update protocol version in KernelInfoHandler
Added endTag method to XmlEmit class
Added steam API initialization and shutdown functionality
Fix testExceed128Connections test to make it compatible with Renjin changes
Improve KieSession API usage in ProcessServiceImpl by enabling signal tracking
Warn when multiple kubeconfig files are found, defaulting to the first one.
Fix testSMILES method, remove obsolete code
Update Conscrypt availability check for Android support
Update query logic in TigerQueryBuilderCanvas to use query value directly from controller state
Simplify getter methods in AbstractPatternRule by removing unnecessary 'return' keywords.
Update BeansXmlDescriptor to use plugin-specific FileDescriptor
Fix: Remove unnecessary initialization of RapidoidInitializer in Conf class
Fixed InputStreamEntity import in MocoPostRequestAction.java
Added AttributeExpression class and attrX method for accessing attribute expressions.
Fix test case to correctly compare tags
Fix scheme method in AbstractTFS class
Optimize decryption of encrypted keys by reducing unnecessary byte allocations
Fix test cases for different class loader and cache manager instances
Fix NullPointerException when shutting down ExecutorService
Fix file path handling in RepositoryPersistenceTest
Update aggregator and collector fragments in ClientInterface to use core classes
Update log level in Log4j configuration
Fixed client interface handle in MpScheduler
Refactor Violation constructor to remove unused code
Add test for inserting a Person with null address and verifying it is selected correctly
Added caching and updating parser method to UADetectorServiceFactory
Added tax ID and exporter identification details for customs declaration test case
Updated AccessLevel enum with new minimal access level
Added support for indentation information in XText formatting plugin
Disable anonymous access for ResourceServerConfiguration
Improve null checks for path when scanning items
Update glue configuration for PostgreSQL tests
Fix mkdirs() behavior and add expectation for creating files in TestDataContextUtils.java
Fixed timing-related statement in DataSetTableIterator class
Added new test classes for Mux and Demux implementations
Fixed opacity setting for raster layers in MapRendererImpl
Improved DockerService: Added validation for Docker machine IP and container network settings.
Fixed nullability annotation on 'create' method
Added null check for connectUri in accept method of ClusterContext
Fix RedisAppender initialization issue by initializing clientName after start() has been called
Layout adjustments for ValidateFrame
Improve retry logic for RemoteRequestRetry to handle timeout exceptions more robustly
Fix test to correctly handle primitive return types in Java Constant Dynamic Test
Fix typo in error message for non-private and non-final classes.
Refactor ProcessInstanceIdCommand to simplify and remove unnecessary annotations
Fix invalid packet increment logic in RetransmitHandler
Fix IndexOutOfBoundsException in Matrix operations
Fix SONARPLUGINS-3097: Ensure Git blame includes last line of input even if it's empty
Added Serializable interface to SimpleReadWriteLock
Improve thread-safety in LifecycleManagerStoreImpl by using putIfAbsent instead of repeated null checks
Fixes bug in CheckConformance implementation by changing index iteration style
Update allowableValues method in ResolvedTypes class to use constant LIST instead of 'string'
Add serialization check for MojoModel
Add event listeners for join/leave and execute operations in DistributedGroup
Update SSH2HostbasedConnect example with password authentication option added
Update SwingTab history management
Update copyright year in ao-dao-base framework
Close PreparedStatement with default database name instead of closing batch
Update Index.java to use consistent method name 'browseFrom' and deprecate 'browseFrow'
Improve SegmentMetadataQueryRunnerFactory by returning a list of SegmentAnalysis results
Update ChemObject.java to use HashMap instead of Hashtable for properties
Fixed warning in SwingUIFacade's showConfirmDialog method by removing unused string concatenation
Fixed enum value comparison in DGLM class
Fix test failure on integration server and add ignore annotation
Fix seed attribute handling for suites and methods
Fix exception handling in ResultExporter, reset image name on band element when an error occurs
Fix typo in ThriftCollection forName method
Fix typo in NotificationsManager.java, add null check for getEmailPath() method
Added JUnit assertion method for floating point equality with default tolerance
Fix typo in Path annotation handling
Added support for time-based metrics in JHipsterMetricsEndpoint
Added delta value for float precision checks in ResizeSettings class
Fix null pointer exception in memory cache handling
Fix invalid ID handling in Message class
Run PackrReduce in place for directory jars
Improve master process configuration for Thrift RPC server shutdown timeout
Update video mosaic example with improved image motion estimation
Fix issue #82 by adding request size validation for large requests
Improve classpath parsing for Windows paths and fix trim logic
Fixed PrettyLogViewerPanel to remove date from simtime string
Fixed status code for concurrency limit error
Improve logging in AddonRepositoryImpl by using fine-level loggers instead of info-level loggers
Added clearViewCached method to allow clearing of cached views
Increase test duration from 20 seconds to 30 seconds
Added flag CACHE_MODE_LOCAL to DistributionManagerImpl
Added box and unbox methods for converting between primitive arrays and Integer object arrays
Fixed logger printing format for stack traces
Added rule for 'PRP_RB_NO_VB' grammar error
Update OperaDriver documentation and description for vendor support and Selenium integration
Update RadarLevel2Servlet to reflect changes in data type and service name
Improve HamcrestCondition constructor and isTrueOrNonNull method for better error handling
Removed unnecessary function library load in OMetadata
Fix TypeVariableCycleTest to correctly handle type parameter cycles with new @TB annotation
Fixed cache size for eager expression factory
Fix NaN handling in Sum node
Added cluster config entry permissions
Fix null listener check in ApnsConnection constructor
Improved finalizer by switching to concurrent map for better performance
Fixed Gaussian proposal distribution sampling value in regression test
Add Transfer-Encoding header for HTTP/2 requests when content is present and backend expects it
Update logic for calculating latest water mark in BaseSource.java
Fixed calculation of next timeout date in TimerTask class
Update SourceVersion enum with new versions (1.7 and 1.8) including diamond syntax and lambda expressions
Add tearDown method to reset ContextHolder after each test
Fix double processing of expressions in ExpressionAnalyzer
Update CheckRange class parameter type fix
Update approvalsRequired property in AsyncMultisubprocessTest
Add Metrics import for HttpClientNoMicrometerTest
Fix dbCreate() method connection issue
Enhance WebXmlObserver to properly handle context class loader changes for improved thread safety and readability.
Fix null pointer exception when creating new ResultPoint instances in GenericMultipleBarcodeReader
Fix parsing of two-digit year values in Japanese calendar format
Added Executor for async execution
Fix serious failure in PolicyFinderModule evaluation of policies
Update assertion percentile in TraversalStrategyPerformanceTest for better performance testing
Fix typo in IHapiTerminologySvc implementation
Fix issue 122: adjust day calendar for date ranges ending at midnight
Fix last available chunk id comparison in GameUpdateTask
Fix racy behavior and add TODOs for additional edge cases in PoolTest.java
Added Vaadin form reference support for AbstractFormPropertyRenderer
Update SdkConfig.java dependencies
Fixed internet connection check in SdkUtils.java to use application context correctly
Fix: skip DIB header when reading metadata from Raf file
Fix span builder to set current span correctly
Fix transient field loading in CRUDDocumentPhysicalTest
Improve extractAsInputSource method by making it final and adding a comment explaining its behavior
Added default constructor and setter methods for ProxyFactoryConfig class
Fixed tree node rendering by setting result index
Update date format in MultipleRelationsAPITest.java
Fix CLI version mismatch check and clarify non-serializable issue with Arrays.asList
Improve ValueComponentHelper by adding support for dynamic input field values
Improve hostname verification in XMPPTCPConnection
Update ReplaceIfSameOperation to record old value when successful
Increase default wait time for InteractiveTestJob from 5 seconds to 10 seconds
Add linear algebraic matrix multiplication method
Fixed termId calculation for correct partitioning in ArchiveConductor
Fix VerifyMojo to correctly handle orphaned documents and add informative error messages
Fix project directory creation in EditorUtilities#copy method
Fixed setCoordinates method to correctly initialize and update geometry
Fix bug in LogReplication#stopReplication method by rethrowing a custom exception instead of ignoring it
Add alternating background color to resources calendar
Fixed datebox modal behavior
Fix bug in BatchLinkingService by passing true to eGet when getting reference value
Update ping interval in SimpleRoundRobinLBTest to ensure proper server initialization
Fixed null pointer exception when combining task deploy and pending task artifacts
Fix test file system path issue in RunLocalPayloadTest
Refactored logging configuration for Server.java
Added page_story_id field to Photo class for published photos
Add locale check for native library loading in AttachContainerCmdExecTest
Added waiting and timeout functionality for TestCommons class
Update account header selection list handling when switching profiles.
Fix redirect URL encoding in RackFilter
Fix null check for registeredService and service parameters in DefaultMultifactorAuthenticationTrustedDeviceBypassEvaluator
Fixed logging in TimeScheduler3 to include task name
Added ConnectionLeaseTimeThresholdExceededEventListener to FlexyPoolConfiguration
Fix null pointer exception in QueryResults.getDeclarations() and add fallback for empty declarations list
Simplify conditions for detecting certain exceptions in PatternTestTools
Fix potential deadlock in ZipExportDelegate by logging warning and including stacktrace with SHRINKWRAP-120
Fix program invocation in ClusterClient
Fix incubating annotation in StackdriverMeterRegistry.java
Update Contents class to use root path for navigation
Fix request content extraction in JEEContext
Added method to resolve resources relative to a consumer class
Add error handling and logging improvements
Fix incorrect bytecode generation for IINC instructions
Added ElasticLoadBalancerCapabilities health check requirements
Fix RedisModuleMemberRegistrationWithTTLConfiguredIT test to use static method on RedisDialect class
Enhance default channel id generation by increasing hex dump padding
Refactor TypeService.java to simplify case logic for ATTRIBUTE_TYPE_GET_OWNERS_*
Update RestHttpHelperTest.java to use InetAddress for host resolution
Fixed tokenizer configuration in WordTokenizer
Add converters and writer exports for TinyLog implementation
Optimize LZ4BlockInputStream compression buffer allocation
Added getter for task speed in Listener4SpeedAssistExtend
Optimize blocking wait strategy by removing unnecessary lock acquisition
Added hint for full redraw after resize
Fix bug where AsynchronousCloseException was caught instead of ClosedChannelException
Fix race conditions in ClassLoaderAdapterCallback when proxying types
Sort collector context values in CollectorContextTest.java
Fix whitespace issues in PRTokeniser.java
Added isInputMethodTarget() method to AbstractTextViewAssert
Refactored platform class with improved injectability and module creation
Update RangeState class to remove transient keyword from Callback field
Update BrowserVersionDetectionTest with fixed method signature
Fix Delegate behavior in MVELLifeCycleManager
Fixed split() method call on line numbers in Parser.java
Update area finder name to use English locale
Added timeout for waiting on stream close notification
Fixed nano formatting in ThreadLog class
Refactor Cycles class constructor with private access modifier
Improve server stop logic by using a centralized executor and security checks
Add JUnit @Ignore annotation for EdgeTest.java
Fix temporary hotfix for boolean comparison of xsd strings.
Fix cyclic vs acyclic graph collation in ApiController
Update PushApplication entity with new imports and UUID type declaration.
Added setter methods for CSV exporter configuration and updated writer initialization with custom separator and quote.
Fix FlowableReverse.reverse() to correctly handle iterable source values
Remove unused registerSubsystemDeploymentWriter method from ApplicationClientConfigurationPersister
Added date-related fields and methods for Cookie class
Fix typo in StreamReaders.java, add support for UNION stream type
Added method `ofBootLayer()` to class file locator interface
Update DeployMojo to reflect changes in external URL creation behavior
Added initialization flag for beta features
Fixed client order ID generation in HitbtcTradeServiceRaw
Added method to retrieve transaction controller in PersistentPriorityProcessorPool
Added WebSocket example with API client usage
Fix CommerceAccountAddressAdminDisplayContext to correctly retrieve commerce addresses
Added ENSURING_CLEAN_SESSION capability for Internet Explorer in DesiredCapabilities
Fixed logging configuration in LoggingAfterJobListener
Fixed JadeTemplateEngine to use 'this' instead of 'model' in get() method
Improve RichFacesTest repeatUntil method to disable waiting for AJAX after determining it must wait.
Fix channel URI in DriverConductor to use IPC channel correctly
Improve performance of ComponentCompaction by using string iteration instead of char iteration for text components.
Update import statements for JAXB and XPath APIs
Added event listener registration functionality
Fix download id generation in DownloadFromUrlInstaller
Fix reading from input stream in SnappyInputStream
Added new method to initialize font icon typeface from a File object
Fix null pointer exception when updating molecular formula's charge
Update ElementUtils with static array for StaxEvent tags
Fix typo in AtomicFileWriter documentation (@since)
Added fast configuration for QR Code detector
Refactor ClassPathBuilder to only locate required codebases when necessary
Fixed trimTimeAnno method to correctly format time strings
Update FoxPlatformDemoDataDeployer to use PlatformService instead of ProcessEngineService
Added null check for User-Agent header in ServiceThemeResolver
Added check for old age and throw BankInterruptedException when robot is too old.
Added Samsung sensor type for orientation detection
Fix bug where log entries were not being added to the work queue correctly in TriggerTask
Improve ShellErrorHandler's handling of stack trace flag by removing unnecessary trimming and type casting
Update tsd.query.allow_simultaneous_duplicates setting to true
Fixed typo in keepAliveTimeout field description
Fix ClusterEventDissector append logic for endpoint field
Improve handling of null InputTypeSummaryResponse values in InputService
Add support for flyWeightTask queue management
Fix bug in SimpleConfigList constructor where it incorrectly accepts invalid resolve statuses
Fix timeout issue in IsStillRunningServiceTest with MultiCallback
Update connection handling for stale sessions
Add runtime configuration for LocalTestEnvironment with optional runtime folder
Fix minor bug in UnsafeBuffer implementation
Fix Conditional Expectation Estimator Interface usage in RandomVariableDifferentiableAAD
Fix error handling in AbstractSingleImageLayer by throwing an exception when content type is invalid
Fixed hedge error distribution print in BlackScholesDeltaHedgedPortfolioTest
Update WlSeat proxy interface to use WlSeatEventsV3
Fix test to correctly handle page fetching and verify no exceptions are thrown
Update shard enum constants with Garena base path
Added getReceivedMessages() method to AbstractReceptionBean class
Enhance loadMetadata method with default LoadMetadataOptions
Improve handling of ClassNotFoundException and LinkageError when loading classes for ServletContainerInitializerRegistrationBean
Simplify TupleClassMeta constructor call
Update version number in ArrayProxy class
Fix null pointer exception when checking reflexivity for saturated chains
Fixed async file write logic to correctly calculate and update committed size
Optimize logic transformer by removing duplicates from AND chains when or is null
Added versioning support for GrailsDomainClassPersistentEntity
Fix batch start awareness in BatchEventProcessor
Fix failing test on conscrypt
Added isMainMethod() check for main method analysis in FindOpenStream detector
Increase audit log queue size to 10000
Fix language comparison in Main class to also consider external languages
Fix socket destination validation in ClientRequestExecutorPoolTest to handle invalid hostnames
Fix error handling in SegmentHTTPApi for non-200 responses
Fix typo in condition name and remove unnecessary backtick
Fix logging of dropped errors in Operator class
Fix null check for uniqueId in Description class
Fix HALBuilderTest: Ensure correct data type in HAL response when executing 'get' on attribute
Added support for multiple place types in NearbySearchRequest
Close WebSocket selector and server on shutdown
Fixed fields method in ODocument class to ensure fields are passed in pairs as name and value.
Remove unnecessary access checks when deleting current links from AccessSetUpdate
Fix select option clicking logic in BrowserTest class
Fix server version parsing in ProtocolConnectionImpl
Update NodeCmd.java to include gossip initialization status in printInfo method
Refactor ShaderSelectFilePropertyHandler to remove unnecessary 'shader/' prefix from shader paths
Fix beacon mac address concatenation in EstimoteBeacons.java
Update calculation for header height based on all-day event count
Added toString method for PostmarkResponse class
Fix bug where association check for http context was not correctly handling allowReAsssociation option
Refactor HetatomImpl to improve naming consistency and atom selection
Fix non-encrypted connection failure when sslMode is PREFER
Fixed issue time parsing in LivecoinAdapters
Enhance profiler logging with configuration details
Improve MapFlatSelectiveStreamReader's block creation for better performance
Remove deprecated methods from DBCollection class
Add sort parameter to JobInstanceFetchRequest for unsorted sorting
Fix exception handling in JerseyServerSentEvents implementation
Fixed AddMenu reAddItem method to prevent adding duplicate items
Fix synchronization issue when checking lock status
Add address operation separator to OperationRequestCompleter
Add option to select download peer manually and improve peer group management
Improve follower state election timeout calculation
Update AWS EC2 version in AWSCloud class
Fix bug in GitRepository#watch method where it did not properly handle exceptions on the executor thread
Fix typo in RollbackTest.java author attribution
Fix test refresh() method in JdbcBaseDaoImplTest
Fix KafkaSupervisor exception handling to catch all Throwable exceptions
Add missing regex to ignore in TFGraphTestZooModels
Updated TargetType enum values for method return types and parameters
Fix bug where FileInputStream was not being created for each next() call on a FileIterator, instead it was always being created from the same iterator value
Update JDBC driver version and major/minor versions for Axibase Time Series Database
Fix: Propagate InterruptedException instead of throwing Throwables.propagate(e)
Refactor PushManager to properly handle notification queue and retry queue.
Add getter methods for ResponseWrapper properties
Added JPA annotation generation option to PropertiesBase
Fix typo in ZipFileUtil.getZipFile() method
Added support for PUT requests in ProxyServlet
Improve rollback handling in ObjectifyImpl by catching and logging exceptions instead of always rolling back.
Fix class name extraction in URLPackageScanner
Fix test to wait for all jobs to complete instead of just 1 job
Fix AzureOSImage name generation for non-standard version indices
Add method to process incoming and outgoing wires
Added special case for Type.T_ANY() in NewJavaFileWriter
Update compiler config constants for compatibility with new byte code generator
Update RuntimeConstants.java to use major.minor format for docs URL
Improve lifecycle handling by adding support for closeable resources
Fix typo in JPerfmonParamsDialog.java, corrected 'initMetricRadios' calls
Fix minor typo and formatting issues in PdfBoxGraphics2D.java
Add null check for _attachments map in UnsavedRevision.java
Fixed font size in SpellAndGrammarCheckDialog
Add test for empty string parsing to ignore
Fix Aesh Bug by removing unnecessary variable name manipulation
Fix bug in successful authentication test for POST requests to protected files
Fix bug in ImmatureClass.java: missing equals method for certain classes
Fix thread shutdown logic for Emitter
Add support for RxJava observables in EmptyInventory
Fix deprecated method in ViewFetcher.java
Add support for Java 8 Deque interface to ImportStack
Fix default configuration behavior in WebSecurityConfigurerAdapter
Fix cluster tests to use latest counter reader API
Added createTempFile method for creating and managing temporary files
Refactor EmployeeDAO to simplify insert methods and remove unused inserts
Update AbstractDSLMappingEntry class to use new getter methods for section and metadata symbols
Fix role loading error handling in UsersResource
Increase maximum processing time for JobServiceJmsIntegrationTest
Add 'onload' event handler transformation for FastArchivalUrlReplayParseEventHandler
Reversed order of stoichiometry elements in formula generation
Refactor ShutdownSignalBarrier to use a latch for synchronization instead of 'this' instance
Fix process builder configuration when using a custom working directory
Fix round robin execute with failover index calculation bug
Added JsonCreator annotation for creating GrokPattern instances
Fix parsing error when missing type or status values in header meta data
Add support for loopback addresses in ParsedInterfaceCriteria
Fix typo in NodeUtil.java, ensure correct error message for not a getprop node
Updated CustomPathResolver with Interceptor annotation
Update build driver configuration to default internal network connection
Refactored MessageUtils class to improve JSON deserialization
Fix typo in exception handling for Exec class
Fix null partition check in KafkaSupervisor
Add Eclipse-specific patching for Lombok AST compatibility
Update ClusteredObservationBus constructor to allow for default configuration by empty string.
Fix potential null chain issue in PeerGroup startUp method
Added support for setting UFS attributes in WebInterfaceGeneralServlet
Fix event firing issue by changing element locator from tagName to id
Fixed logger initialization for IndexedSet class
Fixed Finder.java to handle error codes more consistently
Fix assertion in VoltCompiler to allow class loading from InMemoryJarFile
Update class loading logic in Util.java
Fix mkdir logic in DnsCacheFile constructor
Fixed a method with incorrect native parameters
Refactored Scripting class to use ContextFactory for context creation
Added support for DECIMAL data type in JdbcTypeMapper
Fix bug in LocalTachyonCluster.stopUFS() method
Fix typo in DefaultStompFrame toString method
Added warning for redundant computation in getExpectedLogBelief method
Fix: Removed unnecessary print statement in DiscoverArchiveManifestFilesRuleProvider
Add marker with title and snippet to PlacesSampleActivity
Update synonyms API call to use correct flags when replacing existing synonyms
Added hashCode and equals methods for Resource class
Update statistics tests with Guava's ImmutableList instead of List
Added 'NOT_A_CHAR' constant for Unicode character not present in valid strings.
Update GPS directory tag names for horizontal positioning error
Update poller constructor with deprecation notice and new standalone poller creation method
Add test for spaces in LDAP URL support
Fix testFilters to correctly verify cell timestamps
Fix nullValuesOnly logic in GroupProperty class
Refactor ClientTest.java to use andThenConsume instead of consumeAfter and various atLastConsume calls
Update SpotSceneDirector.java with new cluster change log information and clear cluster method call
Fix testTransactionOptimisticCacheMgmt2Db to remove commented out method
Update MetricOptions class with removed description field and updated import statements
Update low storage space threshold default value
Fix potential NullPointerException in AsyncTcpSocketImpl's write method
Update default servlet version in Metadata class
Fix: Ensure proper cleanup of IE driver instances when testing multiple drivers
Fix GrailsDomainBinder: always set a property as optional
Fixed gallery configuration locale usage in CmsGalleryField class
Update WordListValidatorTest to include full range of characters in VALID_CHARS pattern
Added default group ID for FractionList
Update maximum single upload size limit in SingularityS3UploaderConfiguration
Fix off-heap value holder flush and metadata update issues
Expand UPC-E for purposes of searching
Change '()' to '{}' in BoxedUnit toString method
Added browser wait logic for Jasmine test runner
Add properties for AutoServiceRegistration configuration
Fix header serialization in MessageSerializer
Fix ST_Explode function to correctly identify geometry column(s)
Allow LottieAnimationView to invalidate only the changed region of its drawable
Refactor BlockLockManager initialization and locking mechanism
Fix readCurrentBlockToEnd method to throw IOException for cache write failures
Improve TemporaryBuildsCleaner by only deleting unconnected artifacts
Added shutdown hook to properly terminate the file cache store's executor when stopping
Added service chaining support and worker pool copying for TargetHandler
Fix build timestamp formatting in ExtractorUtils
Fix potential runtime dependency on ServletContext in Sham class
Refactor SparkExecutor to filter out non-existent vertex messages
Fix typo in fetchRole method call
Fix logging configuration for auth requests in DefaultSocializeProvider
Refactor AlternatingController to simplify alignment check and improve readability.
Add typed extern generation check for Google Closure scope aliases
Improve log handling for per session and thread logs
Added title color setting for RoundedLetterView
Fix BluetoothLeScannerCompat to correctly skip scans if mFilters is not empty
Fixed pattern compilation in Element class to make it case-insensitive (temporarily until a new release is available)
Update PerLineTailSplitter class with new implementation.
Fixed sleep duration in AbstractMindmapsEngineTest
Added security context to PartitionCallableRequest for secure callables
Add error handling for internal failures during authentication
Added methods for creating constructor calls with 'this' and 'super' context
Fixed @class attribute resolution in ORecordAttribute class
Update TransactionConfidence.setOverridingTransaction() to automatically set confidence type to DEAD upon becoming newly dead
Refactor LoggingIntrospector to only print logger and handler information
Refactored DefaultIndexRedirect to remove abstraction
Improve OAuth authentication in REST requests by only creating the service when auth is present
Remove unnecessary print statement from SomeSteps constructor
Added onStart and onResume calls in RubotoActivity
Update user creation URL in UserImpl.java
Improve ChromeDevToolsTestBase setup method to explicitly cast and name it 'setUpChrome'
Clear speak service and remove it from dispatcher
Fix typo in method visibility and add null check for system property
Remove unused Google Guava dependency
Fix pom properties handling for ClasspathKieProject
Fix locking issue with posting change notifications
Fix typo in compareTo method of SingularityTaskMetadata class
Added definitionId field and getter to CaseInformation class
Fix parsing exception handling in DefaultMetricService
Fix size and color initialization in ToggleSwitchBase
Update EsMetricsAccessorTest.java to use host instead of container ip for ES client host
Added notification of preference header selection
Fix minor issue with exception handling in Fmrc class
Improved documentation for CacheCreateConfigOperation with caveats and edge cases
Fix bug in TableUtils.clearTable() where DatabaseConnection was being reused after compilation
Remove deprecated method in Closeables class
Fix null pointer exception when setting session in SecurityActions
Added fast approximation for atan2 using a formula from Stack Exchange,
Fix builder creation logic for HDFSDataset class
Added check for persistent timer execution from other instances
Fix StringBuffer vs StringBuilder in convertToHex method
Reorder stream creation for more efficient iteration
Refactor test methods for method selection behavior
Fix compiler bug and ensure proper closure of resources
Update DistributorFlags to use 'reject-unsupported-caps' instead of '--reject-unsupported-caps'
Add functional interfaces for Throwing, including BiConsumer, BiFunction, and BiPredicate variants
Fix resource path handling in BaseClassFinderService
Improved UI thread management in ScanProgressDialog
Refactor OpenALSoundPlayer to support multiple loop modes and gain control
Remove unused client bundle source for publications data
Added Apache License header and copyright notice in PactCommon
Update thread dump formatting for consistency and readability
Fix typo in log statement for getDomainClass method
Refactored cleanup interval calculation to use a mask for more efficient and scalable cleanup logic
Update supported source version range for Java 11 and later annotations
Fixed LSMSolver boundary condition check for no l1 penalty or bounds
Fixed authentication failure handling in DefaultClientAdaptorImpl
Refactored TincanConcurrentQueueHandler to remove unused imports and simplify its usage of LrsObject.
Fix typo in NamedSubResourceSettingBuilder interface method signature
Fix typo in mkdir() call for upload sub-directory creation
Add Jackson annotations for JSON serialization
Added grace period configuration option to TemporaryJobBuilder
Added null checks for controlWindow and currentProduct in TestOperaDriver
Update debug flag and timeout constants for Shippo API
Added isSupervised method and related logic to DeepLearningModel class
Fix assertion for screenshot size comparison in NewSessionTest.java
Added sudo method for executing consumers under admin privileges
Fixed credential reference attribute definition and added volatile keyword to secret field
Added check for fake bug types and improved error handling in BugInstance class
Improve handling of user-defined fields in ControlFile class
Fix indexing logic for container pages in CmsSolrIndex.java
Update test probe builder dependencies
Fix data header term offset calculation and add condition for sending short frames
Added default constructor and additional parameter for Section class in UnitizingStudy.
Fix alignment calculation in LogBufferDescriptor
Fix IOException messages in FileSystem constructor
Fix concurrency issue in cache expiration test by increasing clean time
Enhance WebElementConverter to use CachingRemoteWebElement with correct parent and file detector setting
Fix missing resource exception handling in ConstantsWithLookupProxy
Improve property mapping by using LinkedHashMap for deterministic order
Improve JDBC_PING logging to avoid stack traces when initialization fails
Improve ClasspathUtil#scanClasses by adding a root package check for pkg method
Update DbTester to use AlwaysIncreasingSystem2 for testing
Fix potential buffer overflow when writing arrays to ByteArrayOutputStream
Fix deprecated method and add TODO for removal in Smack 4.5
Update reserved type names in JavaCodeGen class
Fixed logging in SoapClient#marshall method to prevent unnecessary info logs
Fix file path inconsistencies in SAMFileMerger
Fix copyright and version information in SimpleParameterList.java
Fix test FlowRunnerTest2 to wait for thread to finish before asserting
Fix assertion error handling in JSpecRunner
Avoid unnecessary stack traces for ObsoleteVersionException
Update Config interface constants for case folding behavior
Refactor HuobiAdapters to filter out incomplete orders from order history
Update @ToString annotation to include superclass toString in equals generation by default
Fix: Correctly handle file deletion and renaming in offline ChangeLogHistoryService
Added functionality for repeating commands when Enter is pressed repeatedly on arrow keys
Fix InetAddress.getByName() behavior for IPv4 addresses with negative values
Fix bug where removing a single paragraph block from a multi-block element leaves its children orphaned
Improved resource handling in CoverageTest.java for better classpath integrity.
Fix typo in SessionSummary class, correct 'performing' to 'committed'
Fix precision loss in Timebase resample method by rounding result before returning it
Fix typo in update method call, and reorder calls for consistency
Update GradleCli.java to correctly handle GRADLE_COMMAND parameter in CLI input.
Added CPU and memory usage monitoring probes/sensors
Enable or disable 'All Requests' option for injectable headers based on target selection.
Update ChannelThreadExecutorGroup to use ConcurrentHashMap for executorServices
Optimize PathRetractionStrategy to reduce unnecessary barrier checks
Update service name in CreateServicePage.java
Update Context interface to remove direct references to Servlet API and deprecated methods for better encapsulation.
Enhance ProjectConverter.java for better handling of large file conversions
Resolve app name and handle errors when launching an app from RhodesService
Fix leader appointer startup logic to account for replica start actions
Added tryGetDefault method and improved getDefault method documentation
Fixed JDateTime hashCode() method calculation
Update pattern compilation for nullable and non-nullable annotations
Improve CollectionSubject.create method by removing unnecessary generics annotations
Fixed method name typo in ObjectUtils class
Update auth mechanism mapping for DIGEST and BASIC mechanisms
Add setUp and tearDown methods to TestWithStun class for initialization and cleanup of sipStack and userAgent
Add capability for proxy configuration in NewSessionPayload
Improved exception handling in DefaultLogicResult
Added test URL suffix for colon issue (#4)
Added test for assertArrayTypeWithNull being null-safe
Add server role check for view rename test
Fix Ctrl+C handling in CliTableauResultView
Added isRuntime method to ThreadPoolResourceDefinition enum
Update Stopwatch class to use custom Voltpatches version
Fix parsing errors in schema equivalency checker
Update AndroidTestFinder.java to reflect instrumentation test classes to run by InstrumentationTestRunner
Added ResourceAccessor field and getResourceAccessor() method for improved resource management
Simplify ResourceConfigApplier interface by removing unnecessary keyword 'final'
Fixed charset in Loader.loadFile method
Fix: Ensure that an AssertionError is always thrown when testing postconditions with a non-null reference.
Added attributes field to BodyCells model
Added support for watching template and asset folders in BakeWatcher
Update logging level in test to debug
Fix extraction of DSOs in UnpackingSoSource by ensuring directory writable before creating files.
Update image connection timeout check for out-of-sync transport count
Fix bug where sending a leaving stanza could cause occupant information not to reset correctly
Fix item movement when adding sub-items
Fix PinListSync logging to warn instead of error on failure
Update S3_PING protocol to use HTTPS instead of HTTP for signature generation
Update copyright year in RawType.java
Fix onClose method return type
Fixed concurrent map usage in DefaultFileSystemMaster
Fix LRA filter component initialization handling for potential RuntimeExceptions
Fix hidden node issue in NodeInstanceImpl by properly checking the 'hidden' metadata
Update CommandLineParser.java with improved documentation for n-gram rule configuration and added link to relevant wiki page.
Handle UTF-8 encoded strings with special handling for surrogate pairs and replacement characters
Fix bug where DatastoreImpl throws an incorrect exception when handling errors.
Fix CountCommandResult getCount method to correctly cast response value
Deprecate compilation from project files (Legacy syntax support)
Fix permission check for locking resources in CmsLocaleGroupService
Update AbstractPredefinedQuery to remove Nashorn dependency
Fix NullPointerException in JRubyAdapter's printStackTrace method
Fix potential Elasticsearch issue with flushing after VCF importer service call
Update ElasticsearchSinkE2ECaseBase with cluster routing allocation disk threshold setting
Fix null pointer exception in ByteSelectiveStreamReader close method
Fix null class name logging in App.java for Kotlin middlewares
Fixed PropertiesEscapeUtil.java by removing unused imports and updating level constants.
Fix test failure due to missing wait for safe state in Hazelcast 4.x
Fix issue #754: Use time as nonce when API key is used across processes
Fixed ClassWriter constructor to accept ASMSettings.FRAMES as an argument and added a getClassLoader() override to support inheritance of parent class loader.
Refactor shutdown method to ensure repeated requests are handled correctly
Fix connection close logic in LockJDBCConnection
Fix null pointer exception in SurfaceLayerGL by making surface a final field.
Fixed unnecessary SimpleDateFormat usage in FramerManager.tick method
Fix null check for response.getAllHeaders()
Fix ASN.1 decoding of ECKey sequence
Improve HTTP SecurityPreparer by making roles more flexible and easier to configure
Short circuit property evaluation in interpreter
Fix file system creation logic for ClassPathResourceLoader
Improve Mojo configuration handling by allowing listener to modify mojo configuration before calling super.configureComponent().
Fix invalid URL check in VerifierServletTransaction
Trim whitespace from StoryState output stream and absorb newlines when trimming indices
Fix MockTest.java to correctly initialize int1 property
Simplify and improve BranchTemplateServiceImpl by removing unnecessary function and replacing it with a direct call.
Improve resource management in Delombok class
Fix @ClassRule behavior in RepeatTests.java
Update PlaceRegistry unmapPlaceManager method with additional logging and OID use
Improve Period calculation for fixed-month ranges in Chronology-based periods
Simplify optimization flag in RegisteredLookupExtractionFn
Update ServerRemoveHandler to require runtime for removal operation
Fixes bug where varying lens lengths were allowed in secret sharing
Added utility methods for converting camelCase strings to human-readable names.
Added LTHeadMenu and ContextMenuInterceptor for LanguageToolMenus
Fix null check for chains in KeyChainGroup constructor
Fixed constructor to match new format for creating LegacyAddress instances
Update JUnitTestRunner to use NSException instead of JavaLangThrowable
Added AuthoritiesConstants import for role-based access control
Fix null pointer exception in local variable table entry
Added toString method for BootChargingStationCommand class
Added null check for instanceof FileDatastore in isFileFound method
Refactor HelpFlags to use Java 8 features
Added ToolStrip property to FeatureAttributeWindow
Added pre-crawling plugin execution and corrected duplicate code
Fix mkdirs logic for OutputFileMixin.java
Added getFirstHeader and getFirstHeaderValue methods for retrieving HTTP headers and header values respectively
Update DirectLogFetcherTest.java to use writePkg() method instead of write()}
Fixed system exit delay in delayed shutdown
Fix null pointer exception in CollapsingTitleBar background creation
Fix memory registration failure handling in NatRegMrCall
Update Activity.java to match new API schema, renaming activity_type to type
Refactor CampAST.emit() method to improve performance and simplify implementation
Add test for replicated table with coordinator NLJ node inside ORDER BY node
Added sequence number field to CommonLogFields interface
Fix URL scheme validation in BoxBrowser.java
Refactor CanvasTestBase to make it abstract
Update NotificationConsumer interface to remove public access modifier
Added Locale support for field names in DatabaseValidator
Improve disposal logic for AtomicReference of Disposable
Update Streams class to correctly write input streams instead of output streams and vice versa.
Add JPA bootstrap logic to Setup class
Update RxJavaObservableExecutionHook to static field and use it in Observable.create method
Fix shuffle logic for OOB rows in DRF
Fix row count inconsistency in FindAllQuery
Refactor EeClassLoader to simplify class loading and resource finding
Add verbose logging for icon enrichment
Increase send and receive buffers for unicast sockets in LargeState test
Update AndroidSdk.java to use available platforms instead of hardcoded version
Fixed stale element reference exception response code
Refactor ModelUtils to use a more flexible and maintainable method for checking model classes
Fix expiration date calculation for cookies with max-age greater than 1 second
Added private constructor to JAXBUtil class
Refactor URL filtering in BrowserMobProxyTest to use Configuration.baseUrl
Update C3DTextFieldSkin to bind prefWidthProperty with parent's
Improve test annotations for context tests with @SuppressWarnings
Update TransactionRuleBuilder to reflect its new purpose as a transaction rule builder
Update configuration loading mechanism to use system class loader resources
Updated @Remote annotation with @Local on DiameterStackMultiplexerAS7 interface
Generate random terminate token when not provided in request
Added phenotype validation and individual creation retries to Engine configuration
Update REST API to use correct media type for multipart upload
Improve Kafka client configuration refresh logic
Improve LazyPatternBy class by adding toString method and removing redundant synchronized accessor
Update runtime property map for WebAppDatabasePopulator
Fixed version calculation in ClientSession constructor
Fixed random ID generation in SearchResultServlet
Fix testAnimation() method with @Ignore annotation due to known issues.
Add null-safe constructor and return empty list for uninitialized ListItems
Refactor MutableBySettingFieldOfField class to correctly set indirect fields in setter method
Update HTTP request and response imports for Jakarta Servlet API compliance
Generate HGVS code only for SNVs in Variant class
Update Maven repository path in ArchetypeTest.java for consistency with m2.repo system property
Fix PathMatchingSimpleStorageResourcePatternResolver to always add matching resources to the results.
Added snapshot data and received fragment flag fields to JoinProducer
Update ModelObjectWithChildren to include since version information
Fix gap activation logic when no delay is generated
Fix channel parsing in EntityBuilder to handle unsigned longs
Fix null pointer exception when adding table names to graph
Fix invalid Unicode characters in textInPsqlEncoder
Added isInEditMode() check in init method of RippleView class
Refactor Helper class methods for more memory efficiency and readability
Added tree processing callback for TSMeta
Update DeploymentBuilder constructor to use RepositoryService instead of RuntimeService
Add test for exact two queries with no rows and other thread,
Improve SqlgStartStepBarrier generation of traversers
Fix typo in List constructor conditionals
Fix puller to decrement outstanding messages on completion of message handling
fix: corrected log message formatting for low storage warnings
Fix ElementSelectingTest: Update toggle and click methods for correct behavior
Update GradleCompatibilityExtension with new Gradle version in GRADLE_VERSIONS list.
Added SPARQL service description namespace and endpoint property
Fix potential integer overflow in getLargeMemoryClass
Optimize LoadDataChange performance by eliminating unnecessary conditionals
Add support for transient modifiers in AdapterUtil.java
Fixed parameter parsing in PerfMonCollector
Fix ordering description for UsingToStringOrdering
Improve global AST transformation configuration parsing
Fix testMultipleGraphs to use a smaller fixed thread pool
Fix link remote close and detach events in Engine.java
Disable testCreateBucketedTable due to timeout issues
Ensure thread-safe bitmap decoding by acquiring a global lock
Add 'and' method for logical AND operation between Predicates
Fix: Ensure transitions are cloned with their original duration
Fix null column handling in Index.toString()}
Fix null body in finite state machine warning messages
Fix XML parser issues in RSS test cases (#59167)
Back off drain() call to prevent thread stalling and improve responsiveness
Add support for JFM annotations in Lombok FileObjects
Fix year/day factory tests to include leap year cases correctly
Deprecate constructor with multiple parameters and add new constructor with full name components
Simplified IO object reading in EvolutionTest
Added @Ignore annotation to StageTest class and updated import statement
Update test annotation for WanRESTTest to only use @Category(QuickTest.class)
Change flags() method to return a byte instead of a short, and modify flags(short) method to accept a byte parameter.
Fixed user config types in OptimalCECPParameters class
Fixed rendering mode handling in ScheduleRenderer
Synchronize access to thread-local looper creation
Fix AbstractFile.toString to include file separator and name in path
Added null check in append method to prevent unnecessary DOM manipulations and potential errors
Fix flag encoder uniqueness check and add missing check for empty profile names
Convert separator to forward slash '/' in site mojo class
Fix deserialize value caching issue in ClientDelegatingFuture
Added imports for Condition and SelenideElement classes
Added data setter and getter for SessionInfo class
Update CPU and scheduled time calculations in PlanPrinter
Added private constructor to prevent instantiation of CloudSpannerOAuthUtil
Update MySQL writer commands SQL formats for improved compatibility and accuracy
Added support for URI creation from Path in FileSystemSupport interface
Use weak hash map in SlaveReconnectionWork to prevent memory leak
Fixed random number generation in email publisher
Refactor WeaselProgram to move under io.jenetics.ext package
Fix missingResource substring extraction in ModelTest.java
Improve constant time equality checks in BCrypt class
Improve node mapping logic in DocumentGraphMapper for better readability and maintainability
Fix unused variable in command line parsing
Fix default method handle invocation with proxy object
Added underscored ISO code mapping for I18n class
Added check for type parameters in method calls
Fix bug #630111 by properly handling the variable-length header in ClassFileWriter
Fix JoinRequest packet version mismatch logging
Fix DagManager to correctly check for running jobs in DAGs
Fixed shutdown hook registration in NodeService
Fix formatting of LinkedIn identity URL
Update default worker count in WorkerOptions class
Update FormatterMojo.java to include Java exclusion pattern configuration
Added `upgrade()` method to lift security restriction on underlying channel
Added transaction support for Bidirectional and OutBound resource adapters
Fix recipient provider logic for AbstractBuild and Run instances.
Fixed AudioDisconnectEvent firing when re-establishing audio connection
Flip Buffer object in InstallRequest deserialization
Added null check and documentation for join matcher accept method
Fixed source code extraction step sizes
Fix Windows-specific size warning in ChronicleMapBuilder
Refactor HiveMetadata to filter partitions based on predicate
Update LakeBTC exchange specification with corrected SSL URI and host
Added connection timeout and read timeout configuration, and added a commit method call.
Enhance RefreshListener behavior when no ServiceMatcher is provided
Suppress exceptions in ScheduledReporter's run method
Fix calculation of end column in JLanguageTool class
Improved PairBasedRestfulRouter's restful path determination logic, adding a new method isIdElement for better readability and maintainability.
Fix private field naming convention in GoogleCodingConvention
Update WebSocketSubscriptionTransport constructor with new factory parameter and remove deprecated transport parameter
Fixed ViewPropertyDescriptor.newInstance() method to always return null by default, allowing subclasses to override if needed.
Fix typo in error message for no input file name
Fix null adapter initialization in AbstractPreferenceFragment
Fixed sleep interruption logging level from ERROR to DEBUG
Added Jackson annotations for HostIdentifier constructor and getter methods
Fixed extra escape character handling in CSVWriter
Update BasePairParameters class to remove unused field and update structure member variable
Fixed parsing of time field in AbstractResultParser
Refactor normalise() method for performance and readability improvements
Added default artifact type getter for SvnConnectorConfiguration
Refactor CommandManager to use TreeSet for command names
Fix TypeDescription#findVariable() to return UNDEFINED instead of null when type variables are empty.
Add CacheResultUnavailable exception for cache result unavailability
Added Facebook Messenger extension support for WebButton
Fix SessionDataManager to correctly handle checked-in persistence
Improve logging levels for DiscoveryEntryStoreInMemory
Improve performance in AbstractRepository by using Spring Specifications Executor for query counting
Fixed storage clearing in StandardTitanGraph
Fix channel URI formatting in AeronCluster constructor
Fix null check for index names in CFMetaData class
Update testCovtypeGBM method with @Test annotation
Added name property and getter/setter for GeoJsonObject
Added conductor cycle threshold configuration option
Handle RejectedExecutionException when cleaning resources in ClientConnection
Fix typo in domain configuration file existence check
Fix task state tracker registration and clear fork history on initialization
Fix compiler bug and improve close operation for JavacProcessingEnvironment
Update GitHubProvider to handle missing display name
Added readForRecycle method to ByteBuf class for efficient reuse of recycled buffer contents
Fix null pointer check for 'comment' field in MediawikiApi constructor.
Add test for error propagation when not on subscribe
Update copyright year in RepackageMojo.java
Fix closing of DBCursor after iterating over GridFSDBFile collection
Added @Documented annotation to J2ObjCIncompatible class
Added constructor for Meta class with no assigned ID and defaulting to 0L
Fixed bug in ObjFunction.java by correcting float to double for division operations
Update SSL/TLS protocol version in SslTest.java for MariaDB testing
Fix spelling rule for 'unswar' in German language modules
Fix unpackInt method to correctly read bytes from input stream
Update DevelopmentPlugin to use a single VelocityEngine instance
Improve thread configuration for benchmarking
Added managed attributes for slow and very slow invocation thresholds in ServiceStatistics
Fix bug where OpsAgent was set in dummy mode before client interface creation
Cleaned up path traversal in Fingerprinter.java
Add null check for manual provider in Man.java
Enhance zip file handling in Sync plugin, now it can handle both zipping and unzipping of files, and also handles non-zip files by copying them instead
Fix test for user extensions
Added email and subject to Intent when sending email via chooser
Improve constructor for HibernateBundle with multiple entity classes
Add support for handling canned ACLs in S3ProxyHandler
Fix scroll event handling in MapController
Fix XbaseTypeComputer initialization logic
Fix verification transformer for modified operands in ForAll loops
Added missing registerPlugin call and enabled debug version display
Add version information to Transaction output
Fix showMenuItems method to correctly handle text array lengths
Fix null pointer exception in ExceptionProxy#constructExceptionForClass()
Fix import forwarding in ObjectiveCImplementationGenerator
Add warm-up partition call after instance creation in SmallClusterTest
Update Scheduling.java to use boolean values for singleton schedule option
Update dependency changes calculation for upstream culprits
Update exam reactor strategy
Fix scheduling of latency mark timer for more accurate performance monitoring
Fix logging level for optimized selector
Fix test exception handling in TestUtil.java
Add alt text to DropDownImageListEditor for Selenium discrimination
Update cluster node disk usage stats query to include node role
Fix type argument resolution in AbstractBeanDefinitionBuilder
Update VagrantState enum to include provisioning commands for running state
Added ReadOnlyContext notification emission override
Improve AvroUtils.java by adding a more robust comparison for record fields.
Fix error handling in AbstractRemoteCommandObject
Fix empty string check in DocbookBuilder.java for brandFile and commonBrandFile contents
Fix server closing issue in PythonPlanStreamer
Fixed unused methods in ClassDescriptor
Add toString() method for FluxMaterialize
Added exception handling for config loading in AutoReloadingConfiguration
Fix null pointer exception in AbstractMappingStrategy constructor
Update author name in ProductRegisterEndPoint.java
Fixed constructor for PUBREC class
Add test for large date format handling in PactDslJsonBody
Fix: Removed unnecessary print statement from getProcessor method
Improve environment detection for standalone mode and add tests
Fix inconsistency in retrieve behavior and update doc comments accordingly
Update security feature set check in GrpcServerComponent
Add support for IBM's GMT timezone ID and update utc check logic
Refactored NUM_FILES_PER_CHUNK constant with a suggested value of 200 for improved load leveling and reduced overhead.
Add support for monolingual and parallel corpus in LanguageTranslation service
Update NotifyTest.java with new PostgreSQL version check
Fix expired listener removal in PushManager
Added support for FeatureRequirement in AbstractGremlinTest
Fix StrictMode configuration in MainActivity.java
Update SnapshotRestore.java to fix JSON parsing issues and add optional dupsPath parameter
Fixed artifact query location handling and added toString method
Update DiskLruCache documentation to clarify creation vs editing of entries
Fix thread count in AsyncScalr service configuration
Update method signature of afterNodeLoaded() to accept NODE instead of Node
Add serialization of last genomic position scores for raw and scaled values
Added new method to InternalKieBuilder interface for building all KieBases in a module with optional Java source file filtering
Clear attachments from AbstractDeploymentUnitService after stopping sub-deployment
Fix removeAll method to return a boolean indicating whether any elements were removed
Fix bug in JsonSchemaValidationTest where repositories were not being set up with correct schemas
Fix null pointer exception when instantiating LazilyLoadedCtor
Fix election logic to include file sync level in candidate term id calculation
Added conditional for disabling project stats summary
Fixed nullability of user ID in OrganizationDto class
Improve idle detection and window loading logic in OperaDriver
Improve SwissKnife toDate method for date parsing
Fixed RuntimeException handling in Script.java's execute method
Fix bug in PartiallyConstructedObjectAccess detector that incorrectly reported private methods as bugs
Update copyright year in AnalyzerPhase.java
Fix corrupted hint name handling in HintedHandOffManager
Improve IFFImageWriter's progress reporting and correct buffer calculations
Fix SONAR-4245: added path support for resource persister
Updated OPEN_FILES_LIMIT configuration value from 8000 to 1024
Fix backpressure handling in ChannelDuplexHandler
Fixed invalid type checks for sharding properties
Update AndPermission.java to accept Context instead of Fragment for opening permission setting page
Fix test assertion for acquired locks in PidRuntimeFactoryTest
Fix rename operation in MasterInfo class
Added cache for access checks in Permit class
Optimize PBDSegment's truncated entry writing
Fix cancellation logic in SqlStageExecution
Fix bug in ThriftClusterImpl where it was printing CfDef object unnecessarily
Fix bug where operator would not close when all clients were completed or the exchange client was closed.
Updated TreeNode interface to return a boolean for long click events
Fix bug in ConfigException where ValidationFailed could be thrown with an empty list of problems.
Added temp config file handling for brikarIntegrationTest
Update SubResourceSetting.java
Added support for PriorityResource
Fix null pointer exception in FunctionContext class when handling responses
Fix XMLRuleHandler's token correction logic
Update OneShotBenchmark configuration to increase slice value
Add support for DB2 databases in AddPrimaryKeyGenerator
Fix publication logic to correctly advance term IDs and update corresponding term appender indices.
Add Data annotation with new static constructor option and default values
Add checks for joining mode in snapshots
Add conditional URL addition for tools.jar in JspcMojoClassLoader based on compile mode
Improved debug logging in BitcoinSerializer for better performance
Added compareAndSet method for atomic sequence updates
Fix assertion for inserted student in ModelTest.java
Reduce expected refresh count in PeriodicalRefreshStrategyTest
Refactor AsyncExecute to enable environment-specific execution
Update Search UI with new functionality: call updateUI before evaluating fragment
Fixed DefiniteAssignment.java to correctly track method receiver's index.
Update KieControllerSwaggerIntegrationTest to expect multiple possible titles in Swagger documentation
Improve safe mode timeout calculation and simplification
Add test for auto-compaction with special characters
Fix timeslot duration validation in SseBuilder.java
Increase timeouts for tie-breaker and other tests to prevent occasional false positives
Fix access token validation in MPRestClient
Improved filtering of atomic attributes for output in CmdLineAnnotatorUtils
Fix task manager logic for handling partition writers availability
Simplify cache manager test setup by removing redundant wait for cluster to form call
Fix BeanSpec handling of generic types with empty parameter lists
Fix testGetPerformanceNumbers to correctly handle not found performance numbers and empty container cases
Refactor layer state management to improve memory usage and reduce cloning overhead
Fix TODOs in ResolveExpressionTypesPass to improve type resolution correctness
Added new fields to PolicyEvent class for policy rule URL and details.
Fix typo in LogResource.java, adding missing header for delivered text
Added onRequestPermissionsResult method implementation
Add native implementation for Thread.interrupted() method
Improve error handling in RollingUpdateOpFactory for port conflicts.
Update max text length response in ApiV2.java to use server config's anonymous limit
Added deprecated methods for ContentMap, indicating it does not support putting all values or iterating over entries
Improve URI validation in RdfUtils.normalize() to check for newline and tab characters.
Fix whitespace trimming in XWikiXHTMLWhitespaceXMLFilter
