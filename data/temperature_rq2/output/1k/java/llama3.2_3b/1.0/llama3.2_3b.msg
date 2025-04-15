Improve ErrorList's asMap() method to group errors by category with a lazy map initialization
Fixed bug by changing instance variable to protected and added 'final' keyword to it.
Improved logging configuration in DependencyResolutionProcess for debug mode
Added BlockWorkerUtils class with utility methods for Tachyon block workers
Improved database URL generation in H2TestUtil
Updated primary key constraint violation message pattern for MongoDB dialect
Add test for crawling initial URL with CrawljaxException
Fix frame position logging in ReceiverUdpChannelTransport
Fix bug in Util.java by adding a null check when calling `fromStringList` method
Improve handling of private state in ChangelogStateHandleStreamImpl when state registry is not available
Fixed LSMSolver constructor and added hardcoded value for GLM1_rho for testing purposes.
Improve security context management in SpringSecurityWebProcessingContextBuilder
Add constructor for JedisShardInfo with host, name, port, timeout and weight parameters.
Improve loading initial data in StringStringSyncTest by making it a local method and adding a Javadoc comment.
Remove OutputSymbol class from NodeRepresentation
Improve MaterialDataPager behavior by adding page loading and unloading functionality
Fix expected FFDC test for Downgrade Hold Cursors Under XA
Improved JtaTransactionManager rollback logic for better handling of transaction synchronization registry
Fix test commandOverridesDefaultIgnoreExceptions to propagate non-ignored exception correctly
fix HTTP response handling for JSON content
Update SearchDialogAction layout with caption decorations
Optimize capabilities directory by utilizing a concurrent HashMap instead of an in-memory database
Fix test for CgIsaParamTest.java to remove newline character from output string comparison
Update import statements
Fixed import and renamed from non-static to final.
Refactor BinaryStrFVBuilder with added method 'add' and simplified condition for creating feature sets
Update versioning constants
Fixes issue with skipping nested CityModel groups and adding support for parent CityObject references.

Enhance ClientConnectionsShutdown to wait for graceful connection closures and provide better error handling
RefTest extends PresentsTestBase
Fixed | vs ? in archive conductor uri builder construction
Added verbose mode switch for debugging purposes
Improve logging for missing virtual properties in property set
Improved mocking of roles in AbstractMockingTestCase
Update ContainerlessDockerDeployableContainer to resolve main cube from registry
Refactor TypeExtractor to simplify EnclosedExpr handling
Improve idle state detection by terminating extensions first before shutting down threads.
Added array initialization examples for String and List types
Fix Postgres database container exposure
Fix typo in test suite name
Fix date format in DurationBusinessCalendarTest
Fixed variant model to handle null alternate fields properly.
Update VerifyBase class with a modified static user variable and updated BeforeClass method
Fix spinner adapter issue
Fixed sequence name handling for PostgreSQL databases to avoid quoting removal.
Fixed synchronization issue with XML parsing in STIXSchema
Add validation for Short and Byte data types in ColumnValidationReader
Fix leadership election locking in examples/LeaderElectionExample.java
Update Recycler.java by fixing 'currentScraps' vs 'scrap's comparison
Stabilize test suite by adding back removed Cyrillic name in ConfirmTest.java
Fixed resource loading in ApplicationClassLoader to use class loader's own method instead of system's
Enhanced XMPPConnection addingSyncPacketListener and addAsyncPacketListener methods
Update Layer interface documentation regarding origin-driven rotation
Update MongoDBCollection to handle pagination correctly with skipCount option
Fixed formatting in generated Java comments
Added started() method to Shell class
Added test for typing a letter and its expected caret behavior in KeyboardBehaviorTests
Update JNDIManager.java to use rebind instead of bind for sessionID
Fixed serverTracer clearing current span when tracing is enabled
Fixed test case fraction value in DeepLearningProstateTest
Renamed ClientFactory from private class to public class and renamed defaultReceiveMode constant
Fixed dropped frames calculation in animation fragment
Fixed encoding of attachments in AllureAttachAspects
Update OpenJDKRule to check if tools jar can read instead of being file
Add 'nobs' metric to excluded methods
Improve schema info command by adding support for null relationship types and utilizing SchemaTypeResource
Update version number in CharacterChromosome class
Fix ControllableRunner logic for correct initial tick value
Added finally block to ensure client is properly closed in case of an exception
Update test setup for unmanaged Tomcat service
Fix TestIgnorance test to correctly handle listed services with version in the ignore list
Fix string concatenation in DeepDive06ModelViewController.java
Improve ObjectUnderFileSystem condition for creating metadata when recursively listing directory contents.
Added support for visiting fragment spreads in VariableTypesMatchRule
Refactor socket client to improve data transfer efficiency and readability
Fixed HibernateDaoServiceImpl to correctly retrieve maximum SMS app initial ID and return accurate count
Fix cache logic for DeepWater model builder
Fix RequestAuthenticator.java to handle GET requests differently
Fix UI enabled in local testing for AnchoredPredictorTest
Added delay for RealDevice shutdown to prevent hasty termination in InstrumentsCommandLine
Update test file format in JavaFormatterTest.java
Updated IClassWriter interface documentation for correct usage of annotations
Fixed binop handling in CodePrinter.appendOp method
Improve logging for when we determine we are not master
Add literal representation for Percentile series spec
Refactor Expect.java to remove unnecessary code and simplify overridden apply method
Update logger annotation on SiteAwareBuilder.java
Added missing class EnglishWordRepeatRule in Language.java file
Fixed Proto.java to correctly encode boolean values as 64-bit integers instead of bit fields.
Fixed locale property in CmsGwtActionElement HTML output
Fix source info usage for method assignments
Disable sound for low-priority upload notifications
Fixed exception handling in AutoCompleteComboBox.getChoice() method
Update protocol version in KernelInfoHandler
Added support for ending XML tags with rb() function call and new endTag() method.
Added steamworks4j library initialization and native implementation details
Fix testExceed128Connections by adding sink call for output type after eval
Improve ProcessServiceImpl to use a fetchAll flag when retrieving process instances from the KieSession
Log warning instead of debug when multiple kubeconfig files are found.
Fix bug in ReaderFactoryTest.java

Fixed typos in Query update method and added null check for service in getAvailableAnnotationNames method
Fix typo in property accessors for AbstractPatternRule class
Updated FileDescriptor import in BeansXmlDescriptor.java
Initialize admin configuration and reload settings
Fixed InputStreamEntity import
Added AttributeExpression constructor for accessing Groovy properties
Update assertion for command tags case
Fixed method signature in AbstractTFS class for getScheme() method.
Fix byte array size in decryptKey method of Crypt.java to correctly read decrypted data
Add test cases for different class loader scenarios
Fixed NullPointerException when calling shutdownNow on an ExecutorService instance

Fix aggregator and collector fragments in client interface
Update log level in Log class from DEBUG to INFO
Added client interface handle and connection ID to MpScheduler task creation
fix: remove unused constructor parameter in Violation class
Added test case for null key reference in select method
Added caching and updating parser factory method and deprecated original one in UADetectorServiceFactory
Added test data for CustomsDeclarationTest with updated tax ID configuration
Updated AccessLevel enum with new minimal access level
Added support for custom indentation information with @ImplementedBy annotation and a new Default class
Disable anonymous access in ResourceServerConfiguration
Improve ScannerImpl.java by adding null checks for path before setting fileName on FileDescriptor
Update glue configurations for PostgreSQL tests
Fix test file creation
Optimize iteration timeout logging
Added new test classes Muxl and Demuxl for multiplexing
Fixed map renderer opacity logic for both raster and vector layers
Improved DockerService container inspection logic for edge cases
Improve null check in Connectivity#create method
Improve null check in ClusterContext::accept method
Fixed RedisAppender initialization issue
Improve layout of ValidateFrame by removing redundant code
Enhance remote request retry mechanism for handling timeout exceptions
Fix test method invocation for generic types
Fix typo in error message for non-private and final classes
Refactor ProcessInstanceIdCommand interface to remove unused imports and methods.
Fix retransmit handler packet increment logic
Fix index bounds checks for insertColumn and removeColumn operations in Matrix class


Refactor lifecycle manager store initialization
Improve performance by using an index on rule checks
Fixed allowableValues method in ResolvedTypes class to include the correct output type 'LIST' instead of 'string'
Fix MojoModel serialization issue
Added event listeners for join/leave and execute operations
Fixed SSH2HostbasedConnect.java by changing boolean parameter for SshClient's connect method from false to true
Fixed SwingTab history management by removing unnecessary list and updating history
Update copyright year in Encryption class
Fix closePreparedStatement method to properly handle batch operations
Refactor Index.java: simplify and deprecate obsolete method names for browsing index content
Enhance SegmentMetadataQueryRunnerFactory to support concurrent execution and return all segment analysis results in a single list.
Fix Hashtable import in ChemObject class
Update confirmation dialog text for non-local database migrations
Fixed enum value index calculation in DGLM.java
Fix test failure on integration server and add ignore mark for unreliable test
Ignore seed attribute when overridden by user properties and provide default behavior
Fixed exception handling in ResultExporter, ensuring correct image name reset on band element in case of errors.
Fixed enum value lookup for ThriftCollection
Refactor NotificationsManager.java to remove unused annotation
Added JUnit assertion method for floating point comparisons
Improve Path annotation handling in Environment class
Add support for processing CPU and system process metrics from JHipster Meter Registry
Added delta value for float precision handling in ResizeSettings class.
Fixed null pointer exception in setMemoryCache method by moving scheduleUpdate call into a conditional statement
fix null value validation in Message class
Optimize PackrReduce behavior when dealing with jars and directories
Update master process shutdown timeout conversion
Fixed camera calibration parameters for 2D motion estimation in ExampleVideoMosaic
Added validation for very large requests against predefined max request size
Improve classpathStringToArray method to correctly handle Windows paths.
Fix formatting issue in PrettyLogViewerPanel.java
Improve error handling in ConcurrencyLimitServletFilter by using a named status code instead of hardcoding HTTP 429.
Improve logging in AddonRepositoryImpl by using log levels
Add clearViewCached method to AbsViewHolder to efficiently remove cached views
Update test duration for ArchiveRecordingLoadTest
Set cache mode to local for remote lookup invocations
Added box and unbox methods for converting between int array and Integer array.
Added log stack trace to error messages for improved logging quality
Add new rule for 'PRP_RB_NO_VB' cases with suggestion preferences
Updated OperaDriver.java file to better reflect its vendor-supported nature and affiliation with the Selenium project
Update RadarLevel2Servlet to reflect changes in data type and service name
Enhance HamcrestCondition with a new constructor for simpler matches, improving readability and usability.
Fixed loading of function library in OMetadata
Fix TypeVariableCycleTest to resolve type annotation cycle issues
Fix cache size for eager expression configuration
Fixed NaN handling in Sum node
Added new REST permissions for Cluster Config Entry operations
Fix NullPointerException when listener is not provided to ApnsConnection constructor
Add concurrent HashMap to Finalizer class for improved performance
Fixed proposal distribution in LinearRegressionTest
Added content handling for HTTP/2 requests
Fixed Watermark calculation with empty previousWorkUnitStateHighWatermarks
Improve calculation of next timeout date for TimerTask
Update SourceVersion enum with new versions (1.7 and 1.8)
Added tearDown method to reset ContextHolder after each test case
Fixed double processing of ExpressionAnalyzer nodes
Improved type bounds for CheckRange methods to match Java generics
Increase approvalsRequired in AsyncMultisubprocessTest from 2 to 4
Add Metrics class import for HTTP client tests
Improved db creation method in RethinkDBTest
Fix context class loader for WebXmlObserver
Fixed null pointer exception when updating result points with xOffset and yOffset values.
Update JapaneseCalendar test with corrected year parsing for two-digit year format
Add Executor to Async class for improved thread pooling and asynchronous execution
Fix serious failure in PolicyFinderModule policy evaluation
Update assertion percentile in TraversalStrategyPerformanceTest
Fixed condition for handling zero count and total value
Fix issue 122: adjust calendar for correct range end times
Prevent unnecessary task execution when last available chunk id is set in reverse.
Add test for time-to-live (TTL) configurations
Added Vaadin form support for AbstractFormPropertyRenderer
Update SdkConfig.java dependencies
Fixed internet connection check in SdkUtils.java
Fixed incorrect offset calculation in RafMetadataReader when parsing JFIF header
Update SpanBuilder API to set new span as current span correctly
Update CRUDDocumentPhysicalTest to test transient fields correctly after reloading from disk
Refactor XmlExtractorHelper to make it final and consistently use public access modifier
Added default constructor and setter methods for ProxyFactoryConfig class
Fixed tree rendering by setting result index for nodes.
Fix formatting issue in MultipleRelationsAPITest.java
Fix CLI protocol version check and serialization issue with ArrayLists
Fix ValueComponentHelper to correctly retrieve input field value when a value is specified
Enhance hostname verification in XMPPTCPConnection for SSL connections.
Add null check for successful replacement in ReplaceIfSameOperation
Update wait time for InteractiveTestJob
Add linear algebraic matrix multiplication method equivalent to #timesTranspose(double[], double[])
Fixed term calculation in ArchiveConductor to account for query character correctly

Fixed project directory creation in EditorUtilities
Fixed bug in setCoordinates method and updated implementation of compareTo method
Improve error handling in LogReplication when stopping log replication
Add conditional background color to selected days on the calendar
Update date box modal behavior and settings
Improve BatchLinkingService by passing true to eGet() instead of hardcoding its value
Fix ping interval in SimpleRoundRobinLBTest
Update SingularityMesosTaskBuilder to handle empty artifact list in Mesos task deploy
Fix: Update test path to use system property tempDir instead of hardcoded 'liberty.temp.dir'
Update logging configuration
Added page_story_id field to Photo class for published photos
Add locale check for native library loading on Windows and non-Linux platforms
Added waiting and timeout functionality for testing purposes
Update selection list when setting a new account profile
Add support for query parameters in redirect URL
Fix null check for bypass decision in trusted device evaluator

Added ConnectionLeaseTimeThresholdExceededEventListener to FlexyPoolConfiguration test
Improved null checks for declarations in QueryResults class
Refactor test assertion for matching exceptions
Improve error handling in ZipExportDelegate when closing ZIP output stream
Added validation for empty Flink programs and added ProgramInvocationException
Update StackdriverMeterRegistry.java with fix for @Incubating annotation usage
Fixes typo in path selection
Improve JEEContext's getRequestContent() method to handle body extraction more robustly
Added resolvePath method for resolving resources relative to a consumer class.
Add error handling support to RestRouter
Fix typo in bytecode generation for increment operation
Add ElasticLoadBalancerCapabilities health check methods for healthCheckRequiresListener and healthCheckRequiresName
Update RedisModuleMemberRegistrationWithTTLConfiguredIT to use RedisDialect.toBytes method
Fix long value calculation in DefaultChannelId constructor
Fix attribute type owner requests
Fixed host variable to use local IP address instead of hardcoded 'localhost'
Fixed tokenizer to use correct tokenizing characters
Update module-info.java for new converters and writers
Improve LZ4BlockInputStream compression buffer resizing
Added getter method for task speed in Listener4SpeedAssistExtend
Fix blocking wait strategy to allow for early detection of signal needed
Enable full redraw hint for default screen
Fixed bug where NioWorker would fail silently due to missing exception handling
Fix method handle failure in ClassLoaderAdapterCallback
Sort collector context values in CollectorContextTest.java
Fixed whitespace handling when reading characters in PRTokeniser
Add isInputMethodTarget assertion for AbstractTextViewAssert class
Refactored platform initialization, added constancy to injector wrapper and dropwizard module declarations
Fixed transient keyword in RangeState class's mCallback field
Fixed test commands parameter to reduce compilation time
Fix delegate behavior for task lifecycle in MVELLifeCycleManager
Improve section parsing by using string splitting with max occurrences
Fixed locale for area finder property value comparison.
Add timeout for explicit stream check
Fix nano formatting in ThreadLog constructor
Improved constructor signature and added additional parameter
Improve server stop logic by leveraging SecurityActions to retrieve the cluster executor for each cache manager.
Add JUnit @Ignore annotation for EdgeTest.java
Fixed temporary fix for SPARQL_LANG handling, reverted with proper boolean constant creation
Fixed cyclic graph issue in collate endpoint
Update PushApplication model with new imports and UUID support
Added setter methods for separator and quote in CSVExporter class to improve flexibility and readability.
Improve FlowableReverse's flatMap behavior for lists
Remove unused subsystem deployment writer registration method from ApplicationClientConfigurationPersister
Added date-based cookie expiry and getter/setter
Fix typo in StreamReaders constructor for UNION case
Enhance ClassFileLocator with a new method to create a locator for classes in the boot layer.
Update DeployMojo to default off creating external Ingress/Routes by default
Initialize IOSServer with beta feature and simulator status
fix encoding of client order id in HitbtcTradeServiceRaw
Added method to retrieve transaction controller in PersistentPriorityProcessorPool
Add low-level WebSocket example
Refactor CommerceAccountAddressAdminDisplayContext to improve code organization and performance
Add internet explorer browser capability with clean session option
Fixed logging after job listener by removing unnecessary Job name setting in MDC
Fix template engine adapter usage in JadeTemplateEngine
Disable checking for AJAX attributes in RichFacesTest when repeating until condition is met
Fixed channel URI in DriverConductor to use the correct subscribe channel endpoint
Optimize component compaction by using string iteration instead of array iteration for text components
Updated import statements for compatibility and clarity.
Added event listener registration functionality
Update DownloadFromUrlInstaller to use ID instead of downloadable ID
Fix bug in SnappyInputStream where it did not correctly read the remaining chunk from the input stream after a partial read operation.
Added new constructor for FontIconTypefaceHolder allowing initialization with a Java File object instead of just a path.
Fixed null pointer exceptions when checking and updating molecular formula charges
Updated ElementUtils with updated array for StaxEvent tags
Update AtomicFileWriter documentation with temporary path information
Added default fast configuration for a QR Code detector for improved speed
Improve ClassPathBuilder by preventing unnecessary locating of codebases
Fix formatting issue in KWICPanel.trimTimeAnno method
Update process engine service path for FoxPlatformDemoDataDeployer
Add null check for 'User-Agent' header and return default theme if missing
Fix 'old age' check
Added Samsung sensor type for orientation on Android devices
Fix: Add synchronized lock when pushing log entries into work queue
Improve ShellErrorHandler by removing unnecessary string conversion and simplify flag retrieval
Allow simultaneous duplicates in tsd.query
Fixed minor typo in ThreadFactory checkbox label
Add endpoint field to ClusterEventDissector
Improved error handling when fetching input type information
Added new methods for handling flyWeightTask operations in QueueListener
Improve SimpleConfigList constructor to validate resolve status

Update connection logic to handle potential stale connections
Improve LocalTestEnvironment by adding runtime configuration flexibility
Update UnsafeBuffer class hashCode method.
Fixed Conditional Expectation EstimatorInterface usage in RandomVariableDifferentiableAAD implementation
Fix error handling in AbstractSingleImageLayer by throwing a RuntimeException if wrong content type is provided.
Fixed false positive test by removing unneeded print statement

Fix test for fetching a page to ensure no exceptions are thrown on Travis
Update shard constants with Garena base path
Added getReceivedMessages() method to AbstractReceptionBean class
Improve logging in loadMetadata method for better debug output.
Fix bug where classes couldn't be loaded due to LinkageError when using multiple container initializers.
Fixed constructor for TupleClassMeta with reflection service argument simplification
Update version number in ArrayProxy class
Fix test failure due to incorrect assumption about saturation.
Improve async file writing performance
Fix logic transformer duplication issue when chaining operations.
Added versioning support for GrailsDomainClassPersistentEntity, with version and isVersioned methods
Update BatchEventProcessor to correctly handle batch start events.
Fix failing test signersMustHaveCaBitSet on Conscrypt platform
Added isMainMethod check to optimize the analysis of main methods
Increase audit log queue size for AsyncUserAccessAuditLogWriter
Update language comparison in Main class to also consider external language status
Fixed host validation for SocketDestination in ClientRequestExecutorPoolTest
fix: handle error responses correctly when uploading payloads
Fix typo in SpringBootCondition's subclassOnExpressionCondition
Fix logging behavior when dropping unexpected exceptions in Operators
Fix null check for uniqueId in Description constructor
fix: correct HAL response data type in explore HAL test
Added support for multiple place types in NearbySearchRequest#type method
Close WebSocket selector and server connections on shutdown
Improve ODocument#fields method by adding field validation
Fix: Remove unneeded current links to access types without access checks
Improved browser test logic for select options
Fix server version parsing in ProtocolConnectionImpl to correctly set ServerVersionNum attribute
Fix typo in 'Gossip active' output format
Update ShaderSelectFilePropertyHandler to remove 'shader/' prefix from file path labels
Fix beacon mac address concatenation in EstimoteBeacons
Fixed calculation of WeekView header height and update calendar alignment.
Added toString method for PostmarkResponse class
Fix allowReAsssociation bug: prevent concurrent associations for same HTTP context
Improved hetatom implementation by adding more specific search for matching atom when name is long enough.
Fix non-encrypted connection failure when sslMode is PREFER
Update LivecoinAdapters to correctly parse Unix timestamp from issueTime string
Refactor profiler logging to include configuration options
Improve MapFlatSelectiveStreamReader by removing unnecessary null check for DictionaryBlock building
Remove deprecated methods from DBCollection
Add sort functionality to Batch2CoordinatorIT test
Fixed exception handling in JerseyServerSentEvents implementation
Update AddMenu class to use reAddItem method instead of addItem method
Add synchronization to close() method and added a check for closed state
Fixed address operation separator formatting in OperationRequestCompleter
Improve PeerGroup's selectDownloadPeer method to offer customization and flexibility
Improve follower state handling by removing unnecessary last voted for candidate reset
Update AWS EC2 version in AWSCloud class
Improve exception handling in GitRepository#watch method
Fix typo in RollbackTest.java author name
fix: refresh() method correctly updates query cache after an update operation
Improve exception handling in KafkaSupervisor
Update IGNORE_REGEXES with recently added error models for Libnd4j assign #8002 and #7957 issues
Updated TargetType enum values for method return types and parameters
Fix bug where FileInputStream was being created from wrong iterator when trying to load a block file
Update JDBC driver version and major/minor versions for Axibase TSD Driver
Fixed exception propagation in acquire() method of FileSystemContext.
Enhance PushManager by including all unsent notifications in retries and main queue
Enhance ResponseWrapper with delegate access to its methods

Fix typo in testZipFileGetEntriesWithCharset() method
Added support for PUT requests in ProxyServlet
Improved rollback handling in ObjectifyImpl by catching and logging RuntimeExceptions
Fixed URLPackageScanner classpath handling to correctly handle absolute package paths
Improve test for JobManager by using a CountDownLatch
Update AzureOSImage to correctly extract versioned machine image name
Add new method to WireHandler interface
Add special case for Type.T_ANY() in NewJavaFileWriter
Update compiler config constants and add new parameter
Update RuntimeConstants.java to use major.minor format in documentation URL
Improved lifecycle API with additional closeables support
Fix typo in JPerfmonParamsDialog.java and improve logic for populating metric fields
Fixed documentation typo and minor formatting adjustments
Added default initialization for _attachments map in UnsavedRevision class
Fixed font size in SpellAndGrammarCheckDialog
Add test for parsing empty string in `service.parsePath()`
Fix Aesh Bug in ShellImpl by removing incorrect variable name modification
Update test to set GET method on saved request
Fix bug in immature class detection: correctly handle absence of equals and hashCode methods.
Fix issue with Emitter handling thread interruption
Add implementation of Inventory interface for empty inventory cases
Fixed deprecated warning for getScrollListWindowHeight() method
Update ImportStack to use Deque instead of Stack for improved performance
Disable default configuration in WebSecurityConfigurerAdapter
Remove unused code from ClusterTests.java
Added createTempFile method for creating and managing temporary files.
Refactor EmployeeDAO interface to use consistent return type for all insert methods
Refactored `toPatternString()` and `toString()` methods of AbstractDSLMappingEntry class to use symbols instead of sections for consistency and readability
Enhance users resource to handle role loading errors
Increased maximum processing time for JobServiceJmsIntegrationTest
Add 'onload' event handler transformation
Reversed sorting of subunit stoichiometry string builder
Enhance ShutdownSignalBarrier with signalAll() method

Fix round robin indexing issue where it could lead to negative index values
Add JSON creator for GrokPattern class
Improve error handling when parsing metadata from headers
Added support for loopback addresses in ParsedInterfaceCriteria
Enhance NodeUtil.java to check if a 'getprop' is expected for property access nodes
Added @Interceptor annotation to CustomPathResolver and changed priority to application level
Updated build driver configuration to default internal network usage.
Added null check and cast to MessageReceived in MessageUtils.fromJSON()
Fixed evaluation exception handling in Exec class
Add null check for missing topics in KafkaSupervisor
Fix Eclipse-specific issues with LiveInjector
Allow null or empty string for default configuration in ClusteredObservationBus#newChannel
Added null check for chain in PeerGroup.startUp method
Add support for creating root UFS instance in WebInterfaceGeneralServlet
Fix event firing issue by changing element selector
Fixed logging configuration
Fixed Finder class to handle error codes correctly
Fix loading of InMemory Jar files to avoid assert failure
Update classloader usage in loadClass method
Fix DnsCacheFile mkdir error, add non dir directory check and mkdir failed handling
Removed unused Post Effect Shader Manager argument from cullAndRender method

Added support for DECIMAL data type in JdbcTypeMapper class
Refactor LocalTachyonCluster stopping logic
Update DefaultStompFrame toString method to remove unnecessary type qualification
Fix bug in ProjDepTreeFactor where computation of expectedLogBelief was silently returning -10000 instead of pi.
Fixed logging in DiscoverArchiveManifestFilesRuleProvider to prevent stack traces
Added marker options and snippet for places markers on map
Fix typo in batchSynonyms method: added true instead of false for client.postRequest
Added hashCode and equals methods for Resource class
Update tests for stats API with consistent collection types
Added 'NOT_A_CHAR' constant for Unicode character that cannot appear in valid strings.
Fixed typo in GPS horizontal positioning error tag name
Refactor poller methods to remove deprecated usage and provide an alternative method for creating a new poller without context.
Add test for URL with spaces
Fix testFilters to correctly sort cell timestamps by default.
Fix null values handling in GroupProperty class
Improved test completion handling and timeout behavior in ClientTest
Update SpotSceneDirector.java with new cluster change detection log info.
Remove test method from TransactionOptimisticTest
Fix typo in import statement for Description class
Update low storage space threshold default value
Improve performance by calling doWrite() before returning write.
Update default servlet version in Metadata class
Improve InternetExplorerDriverTest by handling driver quits in try-finally block
Update GrailsDomainBinder to always consider properties as optional
Updated gallery configuration handling for locale based path
Update pattern to include acute accent marks in WordListValidatorTest.java
Initial commit of fraction-list module
Updated maximum single upload size in SingularityS3UploaderConfiguration
Fixed memory leak by flushing context after writing back data and added parameter to updateMetadata method to allow for flush operation of writeContext.
Expand UPC-E for purposes of searching
Fix typo in BoxedUnit toString method, return '{}' instead of '()
Add browser wait functionality for Jasmine test runner
Added properties for AutoServiceRegistration configuration and enabled setting, deprecating shouldRegisterManagement method
Fixed intelliJ serialization issue in MessageSerializer
Fixed ST_Explode function to correctly find and extract SRID from first geometry column
Fix invalidation of LottieAnimationView drawable when changing only some frames

Improve error handling for reading from FileInStream
Fix TemporaryBuildsCleaner to only delete disconnected artifacts
Add shutdown of executor in stop method to prevent deadlocks when stopping the cache store
Copy required properties for service chaining and shared worker pool in TargetHandler
Update build timestamp handling in ExtractorUtils
Improve image picker generation by adding servlet context check
Fix issue with SparkExecutor handling non-existent vertices for view incoming messages
Update CommerceAccountUserRelLocalServiceImpl to fetch roles from role service instead of role persistence
Improve logging configuration for HTTP requests
Fix alignment logic in AlternatingController
Added googScopeAliases check in DefaultPassConfig when generating typed externs.
Optimize session log handling by reusing new ArrayList instead of creating a copy
Added color setting for view's title color
Improve BluetoothLeScannerCompat handling of scan results with filters
Fixed pattern compilation in Element class for case insensitive rules
Fixed class declaration of PerLineTailSplitter
Fix sleep duration in test, now aligns with MindmapsEngineServer start time
Add security context to PartitionCallableRequest to create secure callable tasks
Add error handling for internal failures during authentication
Added methods for creating constructor calls in Super and This scopes.
Fix @class attribute handling in ORecordAttribute
Fixed TransactionConfidence logic for newly dead transactions
Enhance LoggingIntrospector to print only logger and handler information
Refactor DefaultIndexRedirect to a concrete class
Enhance OAuth signing for Flickr REST API calls.
Fixed bug where a hardcoded line was printed in constructor of SomeSteps. Removed unnecessary code.
Add call to onStart() and onResume() in RubotoActivity's run method
Fix URL path in user creation API call
Fix typo in setup method of ChromeDevToolsTestBase
Clear speak service and remove it from dispatcher after usage.
Fixed method signature consistency in SeleneseTestCase.java
Fixed import of Google Guava's Sets library in ClientState.java
Fix pom properties handling for ClasspathKieProject to avoid errors when maven pom is not present
Fixed locking issue with posting change notifications
Fixed typo in compareTo method of SingularityTaskMetadata class
Added CaseInformation class with new method to retrieve case definition ID
Added exception handling for metric expression evaluation
Fixed size and color properties in ToggleSwitchBase
Updated EsMetricsAccessorTest.java to use node.getHost() instead of node.getContainerIpAddress()
Fix preference header selection notification
Fix minor issues in Fmrc.java
Add documentation for caching create config operation parameters and behavior
Improve table truncation logic by getting read-write connection directly and caching it
Deprecate and simplify Closeables.closeQuietly method in Guava 16.0.
Fix null pointer exception when cleaning up session listeners in ServletContext
Added fast and approximated implementation of Math.atan2() when FASTMATH option is enabled
Improve HDFSDataset construction by removing unnecessary Builder creation and directly constructing an instance

Fixed StringBuffer bug and replaced with StringBuilder for convertToHex method
Reorder stream construction for more accurate overlap results
Refactor test method to remove unnecessary subclass usage
Fix compiler bug in close() of JavacProcessingEnvironment class
Update DistributorFlags to align with new ConfigValue naming convention for rejecting unsupported capabilities
Added functional interfaces for Throwing interface with predicate and bi-converter variants
Fixed resource path handling in BaseClassFinderService by incorporating package names
Added SwingUtilities.invokeLater for showing progress bar to avoid threading issues
Added new loop methods for OpenALSoundPlayer with support for volume gain
Update Resources interface to remove unused client bundle source
Added copyright header and updated license notice in PactConfiguration.java
Fix formatting of thread dump output to include quotes around thread name
Fix typo in logging statement
Optimize cleanup interval calculation for Whirlpool
Update AbstractInjectAnnotationProcessor to support Java 18
fix: correct bounds check for l1 penalty case with alpha or lambda zero
Handle authentication exceptions properly and reduce retry count after 401 status code
Updated TincanConcurrentQueueHandler to remove unused imports
Fix typo in NamedSubResourceSettingBuilder constructor method
Fixed typo in DomainHttpServer.java, corrected 'mkdir' to 'mkdirs'
Updated ErrorMessage constructor with Jackson annotations for JSON serialization
Added grace period configuration option to TemporaryJobBuilder
Added null checks for controlWindow and currentProduct properties in TestOperaDriver
Update debug flag for Shippo API
Fixed autoencoder flag in DeepLearningModel and added isSupervised method.
Fix assertion logic for image size comparison in NewSessionTest
Added sudo consumer for executing DB operations under admin privilege
Fix bug where credential reference attribute definition was not made static.
Added check for fake bug types and improved error handling in BugInstance class
Fix user-defined field handling in set method of ControlFile
Improved resource type checking for document factory creation
Update test probe builder dependencies and removed deprecated SPI name
Fixed data header and heartbeat buffer clearing
Updated UnitizingStudy class with additional Section constructor and default values
Fixed alignment calculation in LogBufferDescriptor to handle long terms correctly
Fix file:// and hdfs:// scheme errors in URI syntax exception handling
Improve cache expiration test by increasing clean time
Improved WebElementConverter to ensure CachingRemoteWebElement is created with correct context
Improve lookup for constants with missing resource exception handling
Fix property map type in FlatProperties to maintain field order for serialization
Fixed JDBC_PING logging messages to allow for configurable verbosity
Improve ClasspathUtil#scanClasses with package name lookup
Update DbTester to use AlwaysIncreasingSystem2 for testing purposes
Fix potential buffer overflow when writing data from a single-byte array.
Deprecate and modify existing method signature in ConfigureForm class
Update reserved type names in JavaCodeGen class
Improved logging in SoapClient.marshal method to handle disabled tracing
Fix file path inconsistencies in SAMFileMerger class for _SUCCESS file detection and output path comparison
Fixed bug where SimpleParameterList returned zero as count in some functions
Improve test execution flow by using a thread instead of direct invocation
Improve test failure handling in JSpecRunner
Avoid unnecessary stack traces for ObsoleteVersionException
Changed VANILLA constant to false in Config interface
Filter out incomplete orders in HuobiAdapters#convertOpenOrderToLimitOrder
Fix typo in lombok.ToString.callSuper() method annotation explanation
Fix: Delete and rename old file in cleanup method for offline changelog history service
Improve GUIConsole behavior for Enter key press
Update InetAddress byAddress method for IPv4 addressing, add negative values as required in IPv4 address format.
Simplify paragraph block merging by removing parent block
Added InputStream handling for classpath file loading in CoverageTest
Update SessionSummary class to fix typo in COMMITTED state status
Improve precision loss detection and reporting in Timebase.resample method
Update DefaultScaffoldRequestHandler to always update the domain object before binding

Update MonitoringAgentService to include memory usage monitoring and sensor names for cpu_usage and memory_usage probes
Enable or disable 'All Requests' option for injectable headers
Update ChannelThreadExecutorGroup to use ConcurrentHashMap instead of HashMap for executorServices.
Improve PathRetractionStrategy to correctly handle EmptySteps and GlobalChildren in traversal hierarchy
Update service name in CreateServicePage.java
Deprecate direct access to Servlet API in Context interface
Enhance conversion prompts
Add error handling for resolving app name in RhodesService
Fix leader appointer startup block condition for replicas with create start action
Add tryGetDefault method and improve getDefault method's documentation.
Improve JDateTime's hashCode() method by adding a missing date component
Improve nullability pattern matching in ElementUtil.java
Fix typo in static method signature of CollectionSubject.create()
Fixed bug in String comparison logic
Update AuthMechanism for Basic authentication in SecurityRealmService
Add setUp and tearDown methods to TestWithStun to initialize and release resources
Added proxy configuration to NewSessionPayload class
Fixed exception handling in DefaultLogicResult to propagate causes of InvocationTargetException
Added test URL suffix for colon issue (#4)
Added test for asserting array type with null
Add server role check for view test
Fixed Ctrl+C handling in CliTableauResultView to prevent abrupt execution termination
Added isRuntime method to ThreadPoolResourceDefinition enum
Updated Stopwatch import to use VoltPatches namespace
Fix parsing errors in schema equivalent checker
Update AndroidTestFinder.java to correctly identify instrumentation test classes
Added ResourceAccessor field and getResourceAccessor() method to BaseLiquibaseTask for improved resource management.
Simplified ResourceConfigApplier method name
Fixed charset in Loader.loadFile method
Fix: assertThrows should not return null value instead throws IllegalStateException when assertion fails.
Added new fields for column header texts normalization and cell-level attributes to BodyCells model
Added support for watching template and asset folders in addition to content folder
Update logging level in test case
Improve handling of DSO extraction by making the So directory writable and handle potential IO errors when deleting the file
Update image liveness check and send log buffer descriptor
Fix bug where leaving a room could cause stale occupant information in multi-user chat
Fixed issue with adapter iteration when removing items
Improve logging in PinListSync when receiving pinlist fails with exception
Fix S3_PING protocol to use HTTPS instead of HTTP
Update copyright year in RawType.java
Fix close() method in CloseableAdapter to return void directly instead of a Mono
Fixed concurrent map issues in DefaultFileSystemMaster by making them threadsafe with ConcurrentHashMaps
Fix exception handling in LraFilterComponent for potential RuntimeException during bean selection

Improve documentation for confusion rule in CommandLineParser.java
Enhance StringSerializer to better handle UTF-8 encoded strings containing surrogate characters.

Improve CountCommandResult getCount method by using Number instead of Double for 'n' value
Add deprecation warning for compiling from project files
Fixed permission check for locking a resource in locale group service
Refactor AbstractPredefinedQuery to remove unused imports
Fix NullPointerException in JRubyAdapter's printStackTrace method
Fix potential Elasticsearch flushing issue when importing large VCF datasets.
Add Elasticsearch container environment variable to disable disk threshold allocation
Improve memory management in ByteSelectiveStreamReader by setting null all fields at close method
Fix null pointer exception when logging middleware class names in debug mode
Fixed PropertiesEscapeUtil.java by removing unused import and PropertiesEscapeLevel enum reference
Add waitAllForSafeState call after cluster startup in FrozenPartitionTableTest
Improve nonce generation in BTCEBasePollingService for better concurrency support.
Fix bug in ParserTransformer#defineExtendedParserClass by correctly handling the ClassLoader and providing a stable implementation.
Refactor shutdown method to prevent repeat requests and improve reliability
Fixed connection close logic in LockJDBCConnection
Improve SurfaceLayerGL by making its SurfaceGL field final and removing explicit null checks
Simplified time formatting in FramerManager.tick() method by removing unnecessary logic
Improve handling of empty response headers in RestClient
Enhance ECKey decoding to handle unexpected sequence class
Improve HTTP SecurityPreparer by removing unnecessary default values for 'roles' and adding a permit-all option
Add property invariant checking in property method of Interpreter class
Improve handling of file system notifications and file system loading for URI-based resource loaders.
Improve mojo configuration by adding preExecute event handling
Fixed validation logic for invalid input documents
Trim whitespace from story state output stream and newline handling improvements
Add assertion for initial value of getInt1()
Fixed replacement function usage in BranchTemplateServiceImpl
Improve resource management in Delombok class
Fix @ClassRule handling of exceptions in RepeatTests class
Update PlaceRegistry unmapPlaceManager method to log PLOID after removal
Fixed calculation of total months in Period for chronology-based datetime ranges
Optimize lookup extraction fn to always return true if optimize is not set
Updated Stage for remove handler
Fixes inconsistent share length validation
Add utility methods for converting camelCase strings to human-readable names
Add LTHeadMenu and ContextMenuInterceptor instances in LanguageToolMenus constructor
Improve nullability annotations for KeyChainGroup constructor
Improve constructor annotations and add method names for alternative address creation methods
Fix Java exception handling in JUnitTestRunner
Add authorities constants import for user resource
Fix null pointer check in instruction set
Added toString method for BootChargingStationCommand class
Added null check for FileDatastore in isFileFound method to improve functionality and code quality.
Fix typo in HelpFlags.java
Added ToolStrip property to FeatureAttributeWindow
Fix pre-crawling plugin execution order in CrawljaxController
Improve output directory handling in AOT output mixin
Added getFirstHeader and getFirstHeaderValue methods to BasicHttpClient class for easy access to headers and header values.
Fix packet manager channel write method signature typo
Improved system exit delay and added exit message logging in Main class

Fixed memory registration failure handling in NatRegMrCall
Updated activity type property name in Activity.java
Refactor CampAST.emit() to return String instead of StringWriter
Fixed optimization for replicated tables in outer joins with partitioned tables
Added sequencenumber field with default value of Long.MIN_VALUE for serialization compatibility.
Fix protocol scheme validation in BoxBrowser class
Updated CanvasTestBase class to abstract and added static variables
Refactored NotificationConsumer interface with a minor code change
Added Locale import for consistent string case handling
Fix disposal of atomic reference with null values
Fixed Streams class methods to correctly describe input vs output operations with correct block size and closing of streams.
Refactor setup method to reduce duplication and simplify JPA-related code
Update RxJavaObservableExecutionHook for proper creation
Improve shuffling behavior for OOB (Out-of-Bounds) rows in DRF
Improve FindAllQuery's performance by only processing rows that match expected keys
Fix EeClassLoader.findClass() method to correctly delegate to loadClass()
Fix icon enrichment to include verbose logging and a clearer output for no icon file found
Increase send and receive buffers for unicast sockets in LargeState
Update AndroidSdk.java to correctly determine the chosen platform version.
Fixed stale element reference exception response code
Fix ModelUtils class methods to use consistent and correct parameter checking and return types
Update expiration date calculation for max-age cookies
Fix: Added no-arg constructor to JAXBUtil class to satisfy compiler warning
Fix typo in URL filtering in BrowserMobProxyTest.java
Fix invalid TextFieldSkin layout issue by binding prefWidth to textPane's width
Update test methods for context classes with specs and sub-context classes
Refactor TransactionRuleBuilder to align with new rule builder design
Load configuration from system resources instead of classpath.
Added @Local annotation to DiameterStackMultiplexerAS7 interface to enable remote method invocation for EJB 3.0 compatibility
Added default value for terminateToken in AmazonServiceCatalogService
Added phenotype validation and individual creation retries to Engine configuration.
Update REST API to consume multipart/mixed instead of 'multipart/mixed'
Refactor KafkaClient onModification method to refresh configuration on modified servers and groupId properties
Refactor LazyPatternBy class to make toString method publicly accessible and provide synchronized getter for nested By instance.
Update runtime property map for logo image in OMX Web App Database Populator
Fix version in query request builder
Fix random ID generation in SearchResultServlet
Fix testAnimation() method by adding @Ignore annotation for now until it's fixed
Fix typo in Collection type parameter in ListItems constructor and fix not ready assertion
Improve mutability detection for setter methods with nested objects, allowing direct modification of inner fields
Update HttpServletRequest and HttpServletResponse imports for Jakarta Servlet 6.0 compliance
Fix hgvs generation for SNVs
Update home directory path for archetype test case
Fix PathMatchingSimpleStorageResourcePatternResolver to always add matching resources to results
Added field to track whether site has received its first fragment task message
Update ModelObjectWithChildren to include since version information
Fixed gap activation condition when delayed gap generation fails
Fixed signed/unsigned long mismatch in channel retrieval in EntityBuilder
Improve DiffToChangeLog by handling quoted table names in the schema_name reference correctly
Improve text encoding for Unicode characters in DatabaseUtils
Optimize RippleView init method in edit mode
Fix minor bug in Helper class where lineEndings variable was not properly declared within its scope
Enhance TSMeta's tree processing callback for more flexibility and clarity
Enhance DeploymentBuilder API to use RepositoryService instead of RuntimeService
Add test for none rows with other thread exact two queries
Add Traverser generation for SqlgStartStepBarrier step
Fixed List constructor conditionals for list type selection
Improve message handling in Puller by removing unnecessary exception logging and adding more descriptive decrement of outstanding messages.
fix: corrected log message formatting for low storage warnings
fix: click element instead of setting it selected
Update GradleCompatibilityExtension class with the latest available Gradle versions
Added SPARQL service description namespace constants and updated relevant property creation

Update LoadDataChange.java with improved prepared statement handling
Add support for transient modifiers in AdapterUtil.java
Fixed parameter parsing issue in PerfMonCollector
Update UsingToStringOrdering to use natural order of string representation instead of reverse of natural order
Fix typo in disabled global AST transformations configuration
Improve test concurrency by reducing thread pool size
Fix Link Remote Close Event Handling Bug
Fix unit test timeout
Global lock added for bitmap decoding
Improved Not predicate creation
Fix: restore original duration after cloning transitions in TransitionSet.
Improve index string generation by handling null column entries
Fix: Added event body info in logging when event handler is not registered
Improve code coverage by making XML parsing less picky on CI machines
Fix drain() logic in SoundManager.java to avoid thread staleness
Enhance Javac6 JavaFileObject creation with interface and superclass info
Fix year/day loop bounds in TestLocalDate test methods
Added deprecation warning for older constructor and updated documentation
Simplified IO.read and added unnecessary casting removal
Added @Ignore annotation to StageTest class and imported JUnit Ignore test marker
fix: remove parallel test category from WanRESTTest
Change flags method to return byte instead of short, and flag setter to take byte instead of short
Update OptimalCECPParameters to add Boolean configuration type to getUserConfigTypes method
Fixed rendering mode option in ScheduleRenderer class by adding null check with Objects.toString.
Fix synchronization issue in ShadowLooper#myLooper() method
Improve AbstractFile's toString method by including directory separator
Added null check for html parameter in CmsClientReportWidget.append() method
Improve flag encoding handling for uniqueness and validation of existing profiles
Convert separator to forward slash '/' in `SiteMojo.java`
Cache deserialized value in ClientDelegatingFuture
Added imports for Condition and SelenideElement classes
Added setData method to SessionInfo class to set rules bytecode data
Fix CPU time calculation in PlanPrinter
Added private constructor to prevent instantiation of CloudSpannerOAuthUtil
Update MySQL writer commands SQL format strings for new table similarity functionality
Added support for getting file system URIs and improved method naming consistency
Use weak hash map in SlaveReconnectionWork to prevent memory leak
Fix random output file naming issue in ExtendedEmailPublisher.java
Update WeaselProgram class to move from io.jenetics.ext.engine to io.jenetics.ext
Fix missing resource substring extraction in ModelTest.java
Fix constant time hashing for salt and hash in BCrypt comparison
Refactor node handling in DocumentGraphMapper, adding explicit type check

Fix default method handle invocation to correctly access its proxy field
Add underscored ISO code mapping for I18n class
Allow type parameter recovery
Fix bug #630111 by properly handling the read loop for the class file header in JDK7 compatibility mode
Fix JoinRequest packet version mismatch logging
Fixed DagManager hasRunningJobs method to correctly handle empty DAG IDs.
Fixes shutdown hook not being properly registered for graph database shutdown
Update LinkedIn identity creation with ID URL format fix
Added default value for 'workers' option in WorkerOptions class
Added exception handling for Java methods in formatter configuration
Added `upgrade()` method to lift security restrictions on underlying channels
Added transaction support for Bidirectional and OutBound resource adapters in Main class.
Improve CulpritsRecipientProvider.java by removing unnecessary condition check on runResult
Improved WebSocketClient logic for determining voice channels and audio connections
fix: flip buffer in InstallRequest deserialization
Add null check and assert for 'visitor' parameter in JoinMatcher.accept method
Update source code handling to correctly determine end of extraction bounds
Added Windows-specific check for ChronicleMapBuilder configuration
Filter partitions based on constraint predicate when generating table layout for Hive connection
Update LakeBTC exchange specification with corrected SSL URI
Update connection timeouts and add Commit statement to DemoApplication.java
Fix RefreshListener logic to always trigger a refresh even when no serviceMatcher is set
Suppress exceptions in ScheduledReporter's run method
Fix calculation of end column in JLanguageTool.java
Added isIdElement method to determine whether an element in a URL path is an identifier.
Fixed private field naming convention in GoogleCodingConvention
Improve WebSocketSubscriptionTransport constructor by adding required factory parameter
Add default implementation for ViewPropertyDescriptor.newInstance() method
Fix typo in file list intent extras
Improve null safety on AbstractPreferenceFragment creation adapter
Improved logging for sleep interruptions in SQL batches
Added Jackson annotations for JSON serialization and deserialization of HostIdentifier class
Improve CSVWriter handling of extra escape characters
Updated BasePairParameters class to remove unused field and introduce new fields
Improve parsing of date fields in AbstractResultParser
Improve type normalisation by removing unnecessary print statements and allocating data efficiently
Added default artifact type retrieval for SvnConnectorConfiguration
Improved command management by using a sorted set instead of a regular HashSet in getAllCommandNames method
Fix TypeDescription's findVariable method with null handling for empty list of type variables
Add new exception classes for cache result unavailability and deprecation of previous implementations.
Added Facebook Messenger extension support for WebButton class
Fixed SessionDataManager readItem method to always include the session in persistence check
Improve logging levels for DiscoveryEntryStoreInMemory
Improve performance in AbstractRepository by utilizing Spring SpecificationExecutor for efficient data querying.
fix: restore missing clear method in StandardTitanGraph
Update AeronCluster's ingress publication setup for improved code readability
Fix null check for index names in CFMetaData
Fix covtypeGBM test method signature and added annotation for testability
Added 'name' field and corresponding getter/setter methods in GeoJsonObject class
Add Conductor Cycle Threshold configuration option
Fixed connection cleanup handling in ClientConnection when execution fails with RejectedExecutionException
Add existence check for mgmt-users.properties file in AddPropertiesUser class
Update task state and task tracker registration in gobblin runtime
Fix compiler bug in close() behavior of JavacProcessingEnvironment
Fixed GitHubProvider.java to handle cases where displayName is not set, using user.id.id as fallback
Added readForRecycle method to ByteBuf class
Improve null check for 'comment' field in MediawikiApi methods
fix: propagate error through Flux circuit breaker even when error is not on subscribe
Update copyright year in RepackageMojo.java
Improve GridFS cursor handling for exception safety and resource closure.
Added @Documented annotation to J2ObjCIncompatible.java
Added constructor for Meta class with no assigned id and default metadata
Fix bug in sigmoid function calculation
Update SSL test to use DHE RSA with AES-GCM for 128-bit encryption
Fixed spelling rule for German words 'unswer' and added corrections for 'grund auf'
Update DataIO unpackInt method to use correct input parameter name in DataInput interface
Fix DevelopmentPlugin initialization of VelocityEngine and velocityEngines field.
Fix thread configuration for PlannerBenchmarkConfig in AUTO mode
Added managed attributes for slow and very slow invocation thresholds in ServiceStatistics.java
Fix minor formatting issue where stats agent wasn't started before client interface creation
Improved PathTools in Fingerprinter.java to limit search depth and added documentation for cleanup
Add manual provider check in Man.java
Added zip file handling for unzipSync method and corresponding isZipFile method.
Update assertions for extensions in user test
Improved DialogRateMe's Intent sending capabilities by setting email and subject dynamically from resources
Added entity parameter to HibernateBundle constructor for optional registration of additional classes.
Added support for Amazon S3 bucket ACLs when handling PutOptions
Improve map scrolling by notifying listeners of scroll events
Fixed XbaseTypeComputer to correctly handle variable declarations and initialize current scope
Fix issues with modified operands handling in VerificationTransformer class

Added version information to Transaction display
Improve JemmyDSL's showMenuItems method for consistency and error handling
Improve exception proxy construction logging
Fix forward declarations in ObjectiveC implementation generator
Add warm-up of partitions before testing small cluster
Updated Scheduling.java with new singleton option type parameter for ScheduleOption SINGLETON.
Update dependency changes logic after successful build.
Fixed missing imports in WarJSFIntegrationTest.java
Improve latency mark scheduling in StreamSource operator
Refactor logging in SelectorOptimizer to use FINEST log level instead of INFO
Fix testUtil.java to remove unnecessary exception manipulation.
Add alt text to DropDownImageListEditor for Selenium discriminator
Update cluster node role detection in getDiskUsageStats method
Fixed type argument handling in AbstractBeanDefinitionBuilder
Improve VagrantState with additional up command options
Add support for emitting notifications in ReadOnlyContext
Fix AvroUtils to correctly compare record schemas with different field counts

Update DocbookBuilder.java to handle empty brand files more robustly
Fix server closure in PythonPlanStreamer to prevent resource leak when closing process
Fix class descriptor usage count tracking
Add toString() method for FluxMaterialize

Improve error handling in AbstractMappingStrategy constructor
Update author information in ProductRegisterEndPoint.java
fix: added constructor for PUBREC class
Added test for large date format handling in PactDslJsonBody
Fix: remove unnecessary print statement from CollapsePropertiesTest.java
Refactor environment detection for production mode and standalone scenarios
Update ProfileType behavior to return profile type instead of base type in retrieves
Update security feature set check in GrpcServerComponent for Liberty 3.0 and later versions.
Add support for IBM's GMT timezone ID.
Refactored NUM_FILES_PER_CHUNK for improved load leveling and thread overhead
Added binary file support for monolingual and parallel corpora options in LanguageTranslation.java
Update NotifyTest.java with new connection version checks
Fix expired listener removal in PushManager
Add assumeThat statement for asserting supported feature
Fixed StrictMode settings in MainActivity.java
Improved JSON parsing in SnapshotRestore class for better handling of duplicates and additional data
Improved ArtifactQuery toString method and added null check for location field.
Clarify creation vs edit behavior in DiskLruCache
Update AsyncScalr service thread count
Update method signature in FxmlLoadResult to accept a more general Node type
Add serialization of last genomic position scores for raw and scaled values
Added new method to InternalKieBuilder to build all KieBases in a KieModule with a customizable filter
Improve deployment unit service cleanup and attachment removal
Improve removeAll method by returning a boolean indicating whether any elements were changed instead of counting removed ones
Update test setup for JsonSchemaValidationTest to reflect changes in repository setup.
Fixed null reference for LazilyLoadedCtor#buildClassCtor
Improve election handling by updating candidate term ID in cluster mark file after becoming leader
Prevent unnecessary summary creation when 'findbugs.noSummary' is true.
Updated OrganizationDto class with removal of unnecessary fields
Enhance OperaDriver wait logic for idle state detection
Fix typo in SwissKnife.toDate method and improve its handling of invalid input
fix null pointer exception in Script.java's execute method
Fixed PartiallyConstructedObjectAccess detection, ensuring methods are not incorrectly marked as private when they shouldn't be.
Update copyright year in AnalyzerPhase.java
Fix corrupted hint name validation in HintedHandOffManager
Optimize IFFImageWriter by reducing unnecessary progress updates
Fixed ResourcePersister, added path parameter for resource model
Updated OPEN_FILES_LIMIT property with increased value
Improve backpressure handling in ChannelDuplexHandler
Fix incorrect type checking in ShardingProperties
Update AndPermission.java to accept Context instead of Fragment in the createService method
Update assertion in PidRuntimeFactoryTest to include toString method instead of getting acquired locks directly
Fixed rename operation in MasterInfo implementation
Added cache for access checks in Permit#getMethod() method
Add logic for truncated buffers to PBDSegment class.
Fixed cancellation check in SqlStageExecution
Fix bug in ThriftClusterImpl where a CfDef was being printed on debug log
Improve ExchangeOperator handling of closed exchange client
Updated TreeNodeLongClickListener interface with return type changed from void to boolean.
fix: validate error messages in ConfigException
Update temporary configuration handling in exposure test
Fix SubResourceSetting imports for Java 9 compatibility
Added support for PriorityResource class
Fix null pointer exception in FunctionContext class by removing duplicate variables
Fixed XMLRuleHandler.java to improve pattern marking logic and added error handling.
Update benchmark configuration with larger slice value for improved performance
Allow AddPrimaryKeyGenerator on DB2 databases
Fix publication index calculation and cleanup of outdated TermAppenders
Added @Data annotation with new staticConstructor parameter to control constructor generation
Add error handling for snapshots when JOINING operation mode is enabled
Update JspcMojoClassLoader to conditionally add tools.jar URL
Added debug logging for messages sent over network
Added compareAndSet method for atomic sequence value updates.
Fix assertion on inserted student object
Reduce expected refresh count in PeriodicalRefreshStrategyTest

Added call to update user information after evaluating search results
Fix DefiniteAssignment initialStore calculation
Improve swagger integration test to account for multiple title variations in business central documentation
Optimize safe mode duration calculations and timing
Add test for auto-compaction with special characters
Fix SseBuilder validation logic to ensure correct timeslot duration and domain checks
Increase timeouts for tie-breaker and other tests
Fixed null check for access token in MPRestClient getAccessToken method
Added filtering for specific attributes to include in output
Refactor Task.java to simplify check for partition writers availability and correct logic for task termination
Fix test cluster formation logic for multiple cache managers
Fixed BeanSpec handling of generic types with multiple map parameters
Fix test cases for getPerformanceNumbers method with and without PerformanceStateContainer
Optimize memory usage in Layer class
Fix TODOs in ResolveExpressionTypesPass to improve code coverage
Added policy rule URL field to PolicyEvent class
Update LogResource.java to include X-Text-Delivered header
Added onRequestPermissionsResult method to handle Android permission callbacks.
Add native implementation of Thread.interrupted()} method in jre_emul/Classes/java/lang/Thread.java
Add detailed error messages for RollingUpdateOpFactory
Fix typo in getMaxTextLengthAnonymous() method call
Remove deprecated methods from ContentMap class
fix: validate newline and tab characters in URIs
Fixed whitespace trimming in XWikiXHTMLWhitespaceXMLFilter after CDATA block.
