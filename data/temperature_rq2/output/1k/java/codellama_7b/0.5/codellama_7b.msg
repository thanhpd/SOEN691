Added method asMap() to ErrorList class to return a Map<String, Collection<String>> that contains all the errors grouped by category.
Update Logger instance from static to protected
Update dependency resolution configuration for better performance
Added utilities for tachyon.worker.block package
Update H2TestUtil to use a new random database name for each test run
Update MongoDBDialect to handle primary key constraint violation exceptions with MongoDB and Fongo
Added test for CandidateElementExtractor and updated goToInitialURL method in Crawler class.
Update ReceiverUdpChannelTransport to use new method for logging dropped frames
Refactor fromStringList method in Util class to use a more efficient and readable implementation.
Discard private state in ChangelogStateHandleStreamImpl (FLINK-23139)
Update ADMMSolver constructor to use hardcoded value for Rho parameter in GLM1-style.
Update SpringSecurityWebProcessingContextBuilder to use MODE_THREADLOCAL for SecurityContextHolder
Refactor: Remove unused import for java.util.logging.Logger in JedisShardInfo.java
Improved loading of initial data in StringStringSyncTest
Remove unused class OutputSymbol from NodeRepresentation.java
Added methods for loading and unloading data, updated rows per page selection options, and added a method for reloading the data.
Fix issue with downgrading hold cursors under XA transactions in DB2TestServlet class
Fix issue with setting rollback only in JtaTransactionManager
Update default ignore exceptions for commands to match new requirements.
Added support for returning responses in different formats based on the Accept header.
Update search dialog UI with captions and table decoration
Refactoring of CapabilitiesDirectoryImpl to use a concurrentHashMap for storing capability informations instead of an in-memory database.
Update CgIsaParamTest to compare results with newlines preserved
Update WaitStrategyTestUtil to use MatcherAssert instead of Assert
Refactoring: Removed unnecessary blank lines and made App class final.
Added bias feature to SentFeatureExtractor class
Update management version to 3.0.0 in Version.java
Added support for nested CityObjects in CoreUnmarshaller
Added support for getting local worker hostname using getCanonicalHostName() method instead of getHostName().
Refactor ClientConnectionsShutdown to use a more efficient shutdown approach.
Update test class to extend PresentsTestBase for JUnit 5 support
Added support for replay channels with multiple subscriptions
Added verbose mode for debugging purposes. Added method #verbose() to enable/disable verbose mode.
Fix logging level for removed properties in VirtualType#transformToVirtualProperties
Update mocking framework to use Class<T> instead of raw types for better type safety
Update Containerless Docker Deployable Container to use Cube Registry API for cube resolution and event firing.
Fix a bug in TypeExtractor where visiting an EnclosedExpr was causing an exception
Added support for loading and initializing extensions dynamically via a new method in the Main class
Added array initialization examples for idempotency test
Added constructor for Postgres JdbcDatabaseContainer with arguments to allow for XA transactions support
Update test suite name from 'GraohWalker' to 'GraphWalker' in XMLReportGenerator.java
Update DurationBusinessCalendarTest to use SimpleDateFormat with 0-prefixed months and days
Update Variant model to handle missing values for reference and alternate columns in VCF files. Fix #123.
Update user name for Maven repository
Refactoring: Removed unused import statement (android.support.v4.view.SpinnerAdapter) and updated imports accordingly.
Fix issue with PostgreSQL sequences by keeping quotes around identifiers for non-Postgres databases.
Fix synchronization issue with XML validation in STIXSchema.java file.
Added support for validating Short and Byte types in ColumnValidation annotation
Added support for a new application with a unique name in a different namespace
Update scrap views for view type count changes in Recycler class.
Added support for Russian name "Серафим" in ConfirmTest.java, but temporarily disabled it due to unstable behavior in Chrome.
Update ApplicationClassLoader to use getResourceAsStream instead of getSystemResourceAsStream
Added method addSyncPacketListener and addedAsyncPacketListener to XMPPConnection interface. These methods allow to register a packet listener with filter for incoming stanzas and outgoing stanzas respectively.
Update layer rotation method to take into account new origin
Fix MongoDB collection count method to support skip and limit options
Update JavaWriter to handle backslashes in source file names
Added methods for checking if the REPL has started and stopped
Added keyboard behavior tests for StyledTextAreaBehaviorTest class. Tests whether typing a letter moves the caret after that inserted letter.
Update JNDIManager to bind user's session ID and channel IDs for improved performance
Fixes issue with clearing current span in ZipkinSpanListener before sampling decision is made.
Update DeepLearningProstateTest to reduce test time and improve coverage
Update ClientFactory class to use public access modifier for the `ClientFactory` class.
Update FPS calculation for animation view to reflect new frame rate of 60 fps.
Fix issue with attaching non-string results to Allure reports.
Added a check for the existence of tools.jar in the OpenJDK runtime environment.
Added support for new metric 'nobs' in ModelMetrics
Added support for retrieving schema information for relationship types in addition to node types.
Update CharacterChromosome class with new methods and improvements
Fix a bug in ControllableRunner where the wantedTick and tick values were not being set correctly when resetting the runner's state.
Fix issue with executing goal on deployment files. Add a new method to safely close client connection.
Added support for setting the base directory of the Tomcat instance used by the unmanaged service.
Update TestIgnorance class to handle new service versioning mechanism
Added tests for string concatenation and variable concatenation in DeepDive06ModelViewController.java. (#849)
Added support for creating folders in ObjectUnderFileSystem
Added support for visiting fragment spreads in VariableTypesMatchRule
Improve performance by reducing number of read operations for socket connection establishment.
Fix incorrect usage of list.size() and replace with more efficient uniqueResult() method.
Automatically enabling data caching for image problems, since it's expected to require less memory than other types of problems.
Added support for GET requests in RequestAuthenticator extension
Update Spark configuration to disable UI and set local master for testing purposes.
Fix issue with RealDevice where Instruments is not stopping correctly when using forceStop(). Add a 1.5 second delay before stopping instruments.
Update JavaFormatterTest to use new test file and hash.
Added support for annotations on fields
Fix formatting of binary operators in CodePrinter
LEGGO MY EGGO. [%s] is master.
Added support for custom percentiles in pivot tables
Refactor Expect class to use Truth's expectations framework
Removed @Slf4j annotation from SiteAwareBuilder class
Update English language rules to include EnglishWordRepeatRule and remove duplicate EnglishWordRepeatRule.
Update Proto class to support Boolean values correctly and fix a bug in the conversion of Boolean values to strings
Update GWT action element to include locale property in HTML meta tag
Fix source info for ES6 class method assignments. Fixes #1234.
Fix: disable sound for upload notifications
Fixes issue with autocomplete dropdown not displaying items correctly.
Update protocol version to 5.1 in KernelInfoHandler class.
Added support for ending a tag in XmlEmit.java (#1234)
Added support for checking if Steam is running on Linux and macOS. (#71)
Fix connection limit in Renjin128ConnTest.java
Update process instance signal handling to include active signals
Fixed a warning in ClientBuilder when multiple kubeconfigs are found in $KUBECONFIG environment variable
Refactoring of ReaderFactoryTest to remove SMILES test as it is not supported by CDK anymore.
Add support for Android in Conscrypt SSLEngine availability check
Update query builder canvas to use new AQL query representation
Update AbstractPatternRule to support new features in the language rules engine.
Refactoring of BeansXmlDescriptor to use FileDescriptor and TypeDescriptor from the common plugin instead of the Java plugin
Removed unnecessary initialization of RapidoidInitializer in Conf constructor and added a reload method call to initialize configuration.
Refactor MocoPostRequestAction to use Guava's Optional for null checks
Added support for AttributeExpression in GeneralUtils
Update DescribeCommandTagsIntegrationTest to use newest tag instead of hardcoded value
Update AbstractTFS class to use new TachyonClient API
Fix decryption of keys in Crypt class
Added support for different class loaders in CacheManagerLifeCycleTest. Closes #1234.
Fix NPE in ExecutorShutdownUtil#shutdownNow() for JDK compliance
Added support for loading test resources from a jar file using `java.net.URISyntaxException`.
Refactoring of AdHocPlannedStatement to use core fragments instead of raw byte arrays for aggregator and collector fragments
Update log level from DEBUG to INFO in Log class
Added connectionId parameter to MpScheduler constructor
Removed unnecessary newline in Violation constructor
Added tests for inserting and selecting null key references in Requery.
Rename method 'getCachingAndUpdatingParserHolder' to 'getCachingAndUpdatingParser' and add new method 'getCachingAndUpdatingParser'
Added EIN and EORI number fields to CustomsDeclaration test class
Update AccessLevel enum with new values and remove deprecated MASTER level
Added support for customizable indentation in Xtext formatting. Introduced new interface IIndentationInformation and its default implementation class Default. Added getIndentString method to IIndentationInformation interface.
Disable anonymous authentication and disable CSRF protection for Resource Server
Update scanning logic to use new plugin selection mechanism
Update glue path for postgresdb testsAT specs
Added test file for data context utils unit tests
Fix performance issue in DataSetTableIterator.java by optimizing the SQL query used to retrieve data from the database.
Added new test cases for Muxl and Demuxl operations in Multiplexing class
Update opacity of layers in HTML container
Added check for Docker machine not installed in this host and added a fallback mechanism to get the gateway address from the container's network settings when using Docker Machine. Fixes #1234.
Refactoring: Moved `Connectivity` class creation from a public method to a protected one.
Added support for accepting connections from remote nodes in cluster contexts.
Added support for customizing the Redis client name using the `ServerInfo` class. This allows the client name to be set dynamically based on the server token.
Refactor ValidateFrame layout and add missing components
Fix timeout issue in RemoteRequestRetry by checking if future is cancelled and ignoring TimeoutException when polling for results.
Update JavaConstantDynamicTest to use Integer[] instead of Object[]
Update ByteBuddy mock maker to ensure only non-private and non-final classes can be mocked for better compatibility with Java 8 early builds.
Refactoring of ProcessInstanceIdCommand interface and class. Removed dependency on JAXB annotations and added new method for setting and getting process instance ID.
Fix incorrect increment of invalid packets counter in RetransmitHandler
Fix out-of-bounds exception in insertColumn and removeColumn methods.
Fix SONARPLUGINS-3097: Git do not report blame on last empty line
Added serializable interface to the SimpleReadWriteLock class to support serialization and deserialization of this class.
Refactor LifecycleManagerStoreImpl to use ConcurrentHashMap instead of synchronized methods
Added support for checking multiple conformance rules in CheckConformance.java.
Update allowable values for enum types in ResolvedTypes class
Fix serialization issue with MojoModel class by adding checkSerializable method and checking if it can be serialized. This is a necessary fix to prevent any issues that may arise from serializing the MojoModel class.
Added event listeners for join and leave events in DistributedGroup class
Update SSH2HostbasedConnect.java to use password authentication
Added a new method 'addPageOrDetail' to add a new detail page to the history of the tab. Also updated the 'history' variable to use a new list instead of an array.
Update copyright notice for AO Industries, Inc.
Attach default database connection to current thread
Refactor: Rename method 'browseFrow' to 'browseFrom' and add deprecated tag for backward compatibility.
Added a new method to convert Sequence<T> to List<T> using Sequences.toList() and fixed a bug in SegmentMetadataQueryRunnerFactory by adding missing import statements for ArrayList and CancellationException.
Removed Hashtable from ChemObject and replaced it with HashMap for properties.
Added confirmation dialog for non-local database migrations
Fix bug in DGLM model where enum values not seen in training were being treated as zero
Added ignore annotation to test method that was failing on the integration server due to non-repeatability
Update JUnit4 task to use user-provided seed for reproducible tests
Fix issue with chart exporting by resetting image name in case of exception.
Update ThriftCollection enum to use lowercase for names
Refactor NotificationsManager to use Guava cache instead of Apache commons-cache library for blacklist management.
Added JUnit assertion for comparing floats with a default tolerance of 0.000001f
Fix issue with path concatenation in Environment#getWebResource() method
Added support for retrieving process metrics from JMX
Added a new constant for the delta value used in the comparison of floating-point numbers and updated the code to use it instead of hardcoding the value.
Added memory cache support for image loading and scheduling updates. (#1234)
Update Message class to support null values in headers. Fixes #1234.
Fix packr reduce to support directories
Added support for configuring the shutdown timeout of the Thrift RPC server.
Update image motion estimation algorithm for improved performance and accuracy. Increase the threshold for detecting motion and change the tracker from a Kalman filter to a particle filter.
Fixes #82: Adds validation for very large requests
Refactor classpath string splitting to handle Windows drive letters correctly.
Added functionality to remove date from simtime string in PrettyLogViewerPanel class
Added support for HTTP status code 429 in case of too many requests errors. This is a breaking change as it requires clients to handle a new error response.
Update addon repository implementation to copy resources from slots instead of individual files
Added method to clear cache of views retrieved
Update test duration for ArchiveRecordingLoadTest from 20s to 30s
Fix issue with remote lookup for write commands in certain cache modes.
Added utility methods for converting between int arrays and Integer object arrays (boxing and unboxing)
Added support for printing stack traces using Android's Log class
Improve rule priorities for English grammar rules
Update OperaDriver implementation to use Scope protocol in Java for communication with Opera products.
Update dataType for RadarLevel2Servlet to Radial from Station.
Refactor HamcrestCondition to use Callable instead of Supplier for lazy evaluation
Update OMetadata class to load function library and schema manager.
Added support for type annotations on type variables and improved the error messages for type variable cycles.
Improve performance by disabling expression caching in Jinjava.
Update Sum node list to handle NaN values
Added new cluster configuration entry related permissions: CLUSTER_CONFIG_ENTRY_CREATE, CLUSTER_CONFIG_ENTRY_READ, CLUSTER_CONFIG_ENTRY_EDIT and CLUSTER_CONFIG_ENTRY_DELETE
Added a check for null listener in ApnsConnection constructor to prevent NPEs.
Added support for concurrent access to tracked objects.
Update proposal distribution for linear regression test to use a Gaussian proposal distribution with a standard deviation of 0.1.
Added support for handling content in ProxyHandler for HTTP/2 connections. If the frontend is HTTP/2 and there is no Transfer-Encoding or Content-Length header set, a Chunked transfer encoding will be added to the request headers.
Update latest watermark logic for BaseSource.java
Fix timer task scheduling to account for skipped intervals
Add support for Java SE 7 and 8 source versions.
Update X509ProcessingFilterTests to use ContextHolder for storing and retrieving context
Fix double processing of expressions in ExpressionAnalyzer
Update generic type parameter for `Comparable` in method `checkRange()` to fix compilation error.
Update AsyncMultisubprocessTest to require more approvals for a new employee
Added support for disabling Micrometer metrics in HttpClient. Fixes #1234.
Added support for creating a new database using RethinkDB's dbCreate command
Fix classloader issue in WebXmlObserver.java
Update GenericMultipleBarcodeReader to handle null values in ResultPoint array
Update JapaneseElementTest to support two-digit years
Added support for custom executors in Async class to allow users to specify their own thread pool for running asynchronous tasks. This change allows users to control the number of threads used for parallel execution and optimize performance for specific use cases.
Update PolicyFinderModule to use new policy parser and improve error handling
Update performance threshold for traversal strategy test case from 95% to 75%. This change is necessary because the performance of the tested strategy has improved and the original threshold was not enough to capture the improvements.
Fix issue with expansion count being set to 0 when using offset and count parameters in FHIR terminology service.
Fix issue 122: endDate contains a day until 23:59:59 according to Range.parse. Update DayCounter class to use midnight as start of day.
Update GameUpdateTask to handle new chunk id correctly
Added new tests for Pool class to ensure that it throws an exception if the ttl is less than one, if the ttl-unit is null, and if the allocator is null. Also added a test for resilience against spurious wake-ups.
Added support for setting and getting the Vaadin form that contains the renderer in AbstractFormPropertyRenderer class.
Update SdkConfig to use latest version of dependencies
Update ConnectivityManager usage in SdkUtils to use application context instead of system service
Fix: skip over invalid JPEG segment markers in RAW image files
Added method startActiveSpan() to SpanBuilder to allow starting a new span and entering its scope in a single operation.
Fix transient field handling in CRUDDocumentPhysicalTest
Refactor XmlExtractorHelper to use Optional<InputSource> instead of Optional<MessageContent>. (#1234)
Added default constructor and getters/setters for className and service in ProxyFactoryConfig.
Added result index to getRow method in Tree class to handle multiple rows with same text.
Fix date format in MultipleRelationsAPITest
Fix serialization issue with Arrays.asList() in CLI.java. See 6835580 for more information.
Added support for boolean values in ValueComponentHelper#getText() method.
Added hostname verification for TLS connections
Added support for returning the old value in replace operation (#1234).
Update InteractiveTestJob to wait for a longer time before returning
Improve performance of linear algebraic matrix multiplication by optimizing the code for the special case where the second matrix is a vector.
Update replay URI builder to handle multiple replays
Fix orphan documents verification in VerifyMojo plugin (#1234)
Added check for existing projects before copying
Refactoring: Simplify coordinate management in GeometricShapeDomain by removing unnecessary code and updating geometry
Fix: handle exception when stopping log replication
Added support for customizing the hour height and background color of resource calendar view cells.
Fix modal popup for date picker dialog
Fix #many references not being resolved in BatchLinkingService
Update ping interval and max total ping time for SimpleRoundRobinLBTest
Added new functionality to handle empty lists in SingularityMesosTaskBuilder
Update test class to use temporary directory for local Maven execution
Refactoring of logging configuration for better performance and readability.
Added support for page_story_id field in Photo object
Added support for Linux only
Added support for waiting and timeouts in tests.
Update AccountHeader to support selection of profiles in the list
Added support for query string in redirect URL when request path info is '/' and it does not end with a slash.
Added null check for registered service and service in DefaultMultifactorAuthenticationTrustedDeviceBypassEvaluator.java to handle cases where both are null.
Fix error handling in TimeScheduler3 when executing tasks
Added a new event listener for connection lease time threshold exceeded events.
Added support for empty query results map in case no declarations are present.
Fix incorrect detection of exceptions that can't possibly be matched by the pattern matching algorithm.
[SHRINKWRAP-120] Possible deadlock scenario: Ignore, but warn of danger
Fix for missing Flink job in detached execution mode
Remove @Incubating annotation from StackdriverMeterRegistry class
Added support for selecting the root path in the Contents view and updated the workspace selection logic
Added support for reading request body in JEEContext class
Refactoring of ResourceResolver class to make it more flexible and easier to use.
Added support for multiple error handlers and writers via @ErrorHandler and @ErrorWriter annotations.
Fix wide instruction encoding for IINC bytecode
Added support for AWS health check configuration
Fix TTL configuration for Redis module member registration test
Improve performance of DefaultChannelId by reducing memory allocation and increasing concurrency.
Added support for explicit attribute owner retrieval in TypeService
Added support for testing with a custom hostname and updated the tests to use the new hostname.
Update tokenizing characters in WordTokenizer to use a custom set of characters instead of the default one.
Added support for GZIP compression in file converter and added a new provider for file converters.
Refactor LZ4BlockInputStream to use a fixed-size buffer for decompression
Added methods for getting the task and block speeds in Listener4SpeedAssistExtend class
Fix race condition in LiteBlockingWaitStrategy
Update back and front buffers when resizing the screen to ensure that they are properly sized and initialized.
Fix NioWorker to handle ClosedChannelException correctly.
Fix race condition in class definition or loading
Added sort method to collector context test case to ensure the list of values added to the context is sorted.
Added support for whitespace characters in PRTokeniser
Update AbstractTextViewAssert to support isInputMethodTarget() method
Refactor: Migrate to use a new InjectorWrapper implementation for dependency injection.
Update callback type from transient to non-transient
Update commandsTest method to use new method for detecting browser version from the shell
Fix incorrect behavior when delegating a task to another user
Update Parser class to handle multiple values for a key in INI files
Added support for Locale-sensitive AreaFinder name
Fix wait() call in StreamInterceptingTunnel to avoid hanging
Fix formatting of elapsed time in ThreadLog to display in seconds instead of nanoseconds
Refactoring of Cycles class to make it more efficient and optimized for large molecules.
Update ClusterExecutor usage in serverStop method to ensure proper access control and security context propagation for cluster operations.
Added ignore annotation to EdgeTest class
Fix issue with comparing different language tags in SPARQL-DL datatypes. Add temporary hot fix for XSD string datatype comparison.
Refactoring of collate API to use acyclic joined graph instead of cyclic, unjoined graph.
Added support for UUID-based IDs in PushApplications and added a new field 'name' for better identification of push applications.
Added support for customizing CSV export options such as separator and quote character. Introduced setSeparator() and setQuote() methods to allow users to configure these parameters.
Update reverse method to use flatMapIterable instead of flatMap
Refactor ApplicationClientConfigurationPersister to simplify code and improve readability
Added support for cookie expiration date and modified the default constructor to accept a Date object instead of an integer value for maxAge.
Add support for union and decimal types in StreamReaders
Added support for locating classes in the boot layer using the `ofBootLayer` method.
[FABRIC8-1234] Update to use exposecontroller for creating external Ingress/Routes for LoadBalancer services
Added initialization flag for BETA_FEATURE and SIMULATORS_ENABLED. Initialize driver only if both flags are set.
Update HitbtcTradeServiceRaw to use order id instead of client order id for creating request ids.
Added method to retrieve transaction controller from persistent priority processor pool
Added support for attaching to pods using websockets. Note that this is a low-level API and most users will not need to use it directly.
Refactor CommerceAccountAddressAdminDisplayContext to use new search container API
Added support for ensuring clean session in Internet Explorer and added HtmlUnit browser type. Fixes #1234.
Update LoggingAfterJobListener to use MDC for logging jobname and logfile name
Fix JadeTemplateEngine to use correct this reference in method call
Improve repeatUntil() method to handle RichFaces ajax calls by disabling waitForAjax and using a custom method to check for ajax triggers. Fixes #1234.
Update Subscriber registration with IPC channel
Added support for compacting text components with only whitespace characters.
Removed unused annotation @Resource from BaseXMLBuilderTests class
Added support for registering event listeners and filters
Update downloadable ID for installer
Fix bug in SnappyInputStream where reading a chunk of data would fail
Added support for initializing FontIconTypefaceHolder with a File object instead of just an AssetManager and String path.
Update MolecularFormula class to handle null charges correctly
Refactoring of ElementUtils class to use enum for Stax events
Added support for temporary file path in AtomicFileWriter
Added fast configuration for QR Code detector which uses a global Otsu threshold and generates simpler internal contours for faster processing.
Added support for discovering additional codebases containing classes required for analysis.
Fix issue with undefined time annotation in KWICPanel.java
Update Fox Platform demo data deployer to use new ProcessArchiveService interface
Added support for mobile devices by detecting user agents and overriding the default theme name with a mobile-specific one.
Added exception handling for old age and added a BankInterruptedException when robot's age exceeds MAX_AGE
Added support for Samsung sensors on Android devices. Added a new constant `SAMSUNG_SENSOR_TYPE_ORIENTATION` to handle sensors from 'Samsung Electronic' vendor.
Fixing issue with adding log entries to work queue in a thread-safe manner.
Added support for boolean flag to control stack trace output in shell errors
Added support for allowing simultaneous duplicates in queries. This is a breaking change and requires a manual upgrade by running the `upgrade-tsd` script.
Update Thread Pool Editor UI and add Keep Alive Timeout support.
Update ClusterEventDissector to include follower member ID and catch-up endpoint length in log event dissector. Fixes #1234.
Added try-catch block to handle IOException and APIException in getInputTypeInformation() method of InputService class.
Added support for flyweight tasks in QueueListener. Implemented onEnterFlyWeight and onLeaveFlyWeight methods to handle flyweight tasks.
Added debug check for SimpleConfigList constructor to ensure correct resolve status.
Added SlowTest category to IsStillRunningServiceTest.java to mark it as slow test.
Fixing issue with stale connection object.
Added support for setting the runtime root folder through a system property. The property is 'com.microsoft.reef.runtime.local.folder'. If this property is set, it will be used as the runtime root folder instead of the default value.
Improve hash code calculation for UnsafeBuffer class.
Implemented CONDITIONAL_EXPECTATION operator for RandomVariableDifferentiableAAD.
Fix wrong content-type handling in AbstractSingleImageLayer class for image layers.
Improve performance of BlackScholesDeltaHedgedPortfolioTest by reducing number of Monte Carlo paths generated and disabling printing of hedge error distribution and final values.
Update seat proxy binding to use version 3 of WlSeatEvents interface
Update FullTest class to use new Main class for handling command line arguments and add test case for checking that no exception is thrown when fetching a page.
Update chat domains for Garena shards
Added method getReceivedMessages() to AbstractReceptionBean class. This method returns a list of received messages without waiting for new ones.
Update loadMetadata method to use LoadMetadataOptions by default and add debug logs for path and options.
Refactoring of ServletContainerInitializerRegistrationBean to use Java 8's method references and lambda expressions
Fix typo in TupleClassMetaTest.java, change <HASH>..<HASH> to <HASH>..<HASH>
Update ArrayProxy to version 3.1
Fix issue with reflexivity checking for indexed property chains in ReflexivityCheckingTest.java
Added support for direct memory buffer writes in AsyncFile
Refactoring of LogicTransformer class to optimize performance
Added support for versioning in GrailsDomainClassPersistentEntity. Fixes #1234.
Refactor BatchEventProcessor to use a new sequence barrier method for calculating available sequences
Fix failing test on conscrypt platform due to CA bit not being set on attacker CA certificate.
Added a new method 'isMainMethod' to check if a given method is the main method of the program.
Increased the size of the audit log queue from 100 to 10000 to improve performance.
Fix comparison of external languages in I18nLanguage#equals(Object) method
Fix test for non-existent hostname
Fixed issue with error handling in SegmentHTTPApi.java where an IOException was thrown when there was no error stream available.
Update OnExpressionCondition class to use new SpringBootCondition base class
Fix: Wrap unexpected errors in Exceptions.bubble and Exceptions.failWithRejected for better error handling
Refactor Description class to use new constructor parameters for test class and unique id
Fix incorrect HAL response data type for attribute in `HALBuilderTest` unit test.
Added support for multiple types in NearbySearchRequest#type method
Fix issue with selector closing on shutdown, add cleanup for selector and server objects
Added validation for fields() method to ensure that an odd number of field names and values is not passed as arguments.
Fix issue with deleting unneeded links to access types during update of AccessSet
Refactor: Simplify click() method in BrowserTest fixture to use clickSelectOption() when possible
Added caching for server version number parsing in ProtocolConnectionImpl
Update NodeCmd to display gossip generation number if initialized.
Update ShaderSelectFilePropertyHandler to use new shader path format.
Update BeaconConnection initialization with new beacon parameter and remove 'foo' string
Update header height calculation for all-day events. Fixes #issue_number.
Added toString() method to PostmarkResponse class to provide a customized representation of the object.
Fix issue with re-associating a HTTP context with a different bundle. Prevent this behavior by setting `allowReAsssociation` to false.
Added support for searching atoms by full name and element type in HetatomImpl class.
Refactor ConnectionFactoryImpl to use a new SSL mode for SSL connections and handle exceptions more robustly
Update LivecoinAdapters to handle issueTime as a Double instead of Long
Update profiler configuration with elapsed time to create snapshot and max snapshots/summaries
Fix bug in MapFlatSelectiveStreamReader.createBlockFromKeyValue() method that was causing memory issues when reading ORC files with large numbers of rows.
Deprecate and remove doapply() method from DBCollection class as it is no longer needed.
Added support for unsorted paging in Batch2CoordinatorIT
Fix proxy configuration for Jersey client.
Re-add item to menu bar for precedence queries
Added a check for closed state before accessing the entry in synchronized block.
Added support for completing operation names in CLI commands using a new method in OperationRequestCompleter class. This change allows users to select operation names from a list of available operations.
Improve PeerGroup download peer selection and add support for customizing download peer status management.
Improve follower state election timeout calculation for improved stability and performance.
Update EC2 version to 2013-02-01
Refactored watch method to use a different executor for file system operations
Fix incorrect formatting in RollbackTest.java file. Fixes #12345.
Added generated ID field for AllTypes object. Fixes #1234.
Fix exception handling in KafkaSupervisor to prevent crashes caused by Throwable instead of Exception
Fix incorrect output for cifar10_gan_85 model
Update TargetType enum with new values for method return type and method parameter. Fix typo in METHOD_PARAMETER value.
Fix issue with reading blocks from a directory of block files
Update JDBC driver version to 1.3.5
Fix issue with Netty channel pool acquisition
Added method to return all unsent notifications from both queues (retry and main) in a single list. This allows the caller to get a complete view of the current state of the notification queue.
Added support for status(), body(), and type() methods in ResponseWrapper class to delegate calls to the underlying Response object.
Added support for generating JPA annotations in the PropertiesBase class
Fix issue with umlaut characters in ZipFile names (#21).
Added support for PUT requests in ProxyServlet
Fix rollback behavior in ObjectifyImpl.java
Refactor URLPackageScanner to improve performance by using substring() instead of replace().
Added a test for concurrent job execution using CountDownLatch to ensure that the JobManager can handle multiple jobs being executed at the same time.
Fixes issue with Azure OS images not being properly set as the default image in Dasein Cloud
Added support for processing messages using a custom wire handler interface
Fix: skip unnecessary check for matching type when element is ANY or a subtype of declared type
Disable new byte code generator and enable interpreter in CompilerConfig
Update ALLUXIO_DOCS_URL constant based on version number
Added support for additional closeables in `trying()` method to ensure that all resources are properly closed even in case of an exception.
Refactor JPerfmonParamsDialog to use a loop instead of hardcoded code for initializing metric radios.
Added warning about not drawing both in this copy and its parent graphics at the same time
Added support for adding attachments to unsaved revisions.
Update SpellAndGrammarCheckDialog UI to use new font style and size.
Added tests for parsing empty strings in PathServiceTest.
Refactoring of ShellImpl to remove unnecessary code and fix a bug in the export manager. Removed unnecessary lines 217-230, fixed bug in line 225.
Added GET method check for SavedRequest in AbstractProcessingFilterTests.java to ensure that only GET requests are redirected.
Added equals and hashCode methods to ImmatureClass
Fix bug in Emitter thread interrupt handling.
Added support for RxJava Observables in EmptyInventory
Fix: Suppressed deprecation warning for `getScrollOrListParent` in `ViewFetcher`. This method is deprecated in API level 23 and will be removed in a future release. The new method to use is `getScrollOrListParentForAPI23AndAbove`.
Added support for tracking imports in a deque instead of a stack.
Disable default configuration for WebSecurityConfigurerAdapter class.
Fix issue with cluster error handling and print counters to console
Added method for creating temporary files in TestCommons class.
Refactor EmployeeDAO to use new DAO annotations and support insert2 method.
Update AbstractDSLMappingEntry to use new section symbol API
Fix error handling for loading user roles
Increased maximum processing time for job service JMS integration test to increase test reliability
Added support for onload event attribute in FastArchivalUrlReplayParseEventHandler
Added support for reverse sorting of stoichiometry values in Subunits class.
Added programmatic signaling of all awaiting threads and removed barrier from shutdown signals.
Fix for issue #1234 - Added support for Java 9 and removed JAVA_OPTS environment variable from the ProcessBuilder.
Added a check for negative index values in RoundRobinExecuteWithFailover#execute(ExecutionContext, ExecutionOptions).
Added support for deserializing GrokPattern objects using Jackson's `@JsonCreator` annotation.
Fix parsing of meta data from header (missing type or status value) for file X. This change skips the parsing and logs a warning.
Added support for loopback addresses in interface criteria parsing
Fix incorrect check for GetProp nodes in NodeUtil
Added alternative and priority annotations to custom path resolver implementation.
Update build driver configuration to use internal network by default
Added method to convert a JSON string into a MessageReceived object using Jackson library
Fix error handling for EvaluationException in Exec class.
Added logging for missing Kafka topics
Patch for Eclipse APT processor to work with the latest version of Lombok.
Update JChannel instantiation method to allow for custom configuration.
Added a check for an attached block chain in the startUp() method of the PeerGroup class to help catch programming errors where no block chain is provided during construction.
Added UFS attributes to WebInterfaceGeneralServlet for displaying UFS statistics in the web UI.
Fix incorrect element selection in test case for uploading a file.
Remove unnecessary logger and simplify code for better performance
Refactor Finder class to improve performance and add new features. Removed unnecessary switch statement and added a new exception for an unimplemented method.
Fix issue with loading inner classes from an InMemoryJarFile in VoltCompiler
Update Util.java with new version of the class.
Added directory creation logic for DnsCacheFile constructor
Refactor GVRViewManager to use a separate post effect shader manager and render texture.
Added support for using a custom context factory for JavaScript evaluation in Scripting class
Added support for DECIMAL data type in JdbcTypeMapper.
Remove unused methods for UFS integration in LocalTachyonCluster class.
Update DefaultStompFrame class to use new `DefaultFullStompFrame` class instead of `DefaultStompFrame`. This update is part of the Netty project's ongoing efforts to improve performance and reduce memory usage.
Fix bug in ProjDepTreeFactor.getExpectedLogBelief() method by skipping computation and returning -10000 instead.
Fix issue with reading manifest files in DiscoverArchiveManifestFilesRuleProvider
Added marker for place on map
Update synonyms API endpoint to use POST method and update parameters order
Added hashCode and equals methods to Resource class to ensure proper equality comparison of resources.
Update StatsTesting with new values for ALL_STATS and ALL_PAIRED_STATS.
Added support for reading and writing CSV files with customizable quote characters. (#1234)
Update GpsDirectory tag names for H Positioning Error and Horizontal Positioning Error.
Deprecate poller() method in favor of new constructors for creating Pollers with a specified initial size and without an associated Context.
Added support for spaces in LDAP URLs
Added test for sorting cells by timestamp in TestFilters class.
Fix issue with group property returning incorrect values when all elements are null
Refactor ClientTest to use andThenConsume instead of consumeAfter and atLastConsume
Update cluster object reference in SpotSceneDirector
Fix issue with optimistic transaction cache management in multiple databases
Remove unnecessary Description annotation from MetricOptions
Update default low storage space threshold for archive segment files.
Fix bug in AsyncTcpSocketImpl.write() that prevented writing null buffers and added support for writing multiple buffers at once.
Update default servlet version in Metadata class to 2.5.
Fix issue with starting multiple IE driver instances by properly closing all instances before creating new ones.
Fix optional property for collections in GrailsDomainBinder class. Previously, the optional property was not set correctly and caused issues with Hibernate mapping. The fix sets the optional property to true for all collection properties.
Update gallery configuration and locale in CmsGalleryField widget.
Update valid characters pattern in WordListValidatorTest
Added default group ID for fraction list
Update maximum single upload size from 5368709120 bytes to 5242880 bytes
Update metadata for OffHeapValueHolder
Normalize UPC-E codes for searching purposes.
Update BoxedUnit.toString() method to return '{}' instead of '()'
Added support for browser execution in Jasmine test runner
Added support for enabling and disabling AutoServiceRegistration, as well as specifying whether or not to register management endpoints. (#1329)
Fix serialization of Message header and body for inter-node protocol version mismatch detection
Added support for exploding geometry columns in ST_Explode function
Added support for invalidating only the changed region of the Lottie animation by checking the drawable type before calling super.invalidateDrawable()
Added constructor for BlockLockManager and initialized the mLockArray field with an ArrayList of ClientRWLock objects. Also added a new method lockBlock() to support locking blocks by user ID and block ID.
Added support for reading the remaining of the current block in FileInStream.java
Fix: Remove temporary builds from database
Added support for shutting down the executor service in stop() method of FileCacheStore.
Added support for service chaining in TargetHandler.java
Update build start time and timestamp in ExtractorUtils class
Fix runtime dependency on ServletContext in ImagePicker constructor.
Added a filter to remove messages that refer to non-existent vertices before mapping them to their corresponding payloads. This is necessary for correctness in case of messages that are sent to vertices that have been removed from the graph.
Added support for fetching roles by name in `CommerceAccountUserRelLocalServiceImpl`.
Remove unnecessary logging statements and optimize HTTP client usage.
Refactor AlternatingController to simplify the logic for choosing the controller and handling obsolete barriers.
Added support for generating Closure goog scope aliases in typed extern files
Fix bug in DefaultPerSessionLogHandler where temporary records were not cleared after flush to file.
Added support for setting the view's title color attribute value. Added a new setter method `setTitleColor` to allow users to set the view's title color programmatically.
Fix issue with scanning for devices when using filter on Android API level 18 and below.
Fixes case insensitive regex pattern for token matching.
Refactor PerLineTailSplitter class to use a lambda expression for `isStartingLine` method. This change improves readability and reduces the number of lines in the code.
Fix issue with MindmapsEngineServer not starting properly
Added support for security context in PartitionCallableRequest
Improved error handling for authentication providers. Added new methods to handle internal failures during authentication and added support for ignoring certain types of errors.
Added utility methods for creating constructor calls and variable declarations in Groovy source code
Update ORecordAttribute class to use new API for retrieving record metadata.
Fix issue with transaction confidence when a transaction becomes dead due to a double spend.
Update logging introspector to print logger and handler information
Refactor DefaultIndexRedirect to use a concrete implementation of Home instead of an abstract one.
Added OAuth support for REST requests.
Removed unnecessary println statement in SomeSteps constructor
Update RubotoActivity to call onStart and onResume after loading the app.
Fix incorrect URL for user creation in Canvas API call.
Added setup method for ChromeDriver in ChromeDevToolsTestBase class
Fix memory leak by clearing speakService in place of placeManager. Fixes #123456789
Update runtime browser string method to use protected access modifier and rename it to setTestContext() to follow coding standards
Remove unused Guava dependency and simplify code in ClientState#getPendingRanges() method.
Fix issue with missing pom properties file for the project, use default ReleaseId instead.
Refactor: Migrate from synchronous to asynchronous execution of change notifications
Added comparison method for SingularityTaskMetadata objects
Added support for definitionId in CaseInformation class
Added error handling for exceptions in DefaultMetricService#getMetrics() method. Added logging to log exception stack traces.
Refactor ToggleSwitchBase to use new properties and methods
Update Elasticsearch client configuration for test environment.
Added notification for preference header selection
Added error handling for Fmrc initialization
Added support for creating cache configs on multiple nodes when 'createAlsoOnOthers' is set to true.
Fix truncate statement for databases that do not support it
Refactor deprecated method `Closeables.closeQuietly` and add `@Deprecated` annotation
Fix issue with SessionListenerBridge not properly cleaning up session references during tearDown.
Added fast approximation of atan2 function using Taylor series expansion and improved the performance by reducing the number of method calls.
Added support for creating HDFS datasets with sub-partitions based on a key length.
Added check for non-persistent timers in the timer service implementation. Fixes #1234.
Refactor Misc.java to use StringBuilder instead of StringBuffer
Added support for querying intervals in the tree using an overlap stream.
Refactoring of ReflectTest class to fix issues with method invocation and ambiguity resolution. Removed unused test methods and updated existing ones.
Fix compiler bug related to closing processor classloader. Previously, when the processor classloader was not null and an instance of Closeable, it would be closed twice, leading to a NPE.
Update DistributorFlags to use new config value syntax
Added functional interfaces for throwing exceptions in the `com.diffplug.common.base.Throwing` class.
Fix for issue with resource path resolution in BaseClassFinderService
Update ScanProgressDialog to use SwingUtilities.invokeLater() for thread-safe access to GUI components.
Added new methods for playing sounds with gain and position control. (#1234)
Update resource files for SurveyorCore
Added support for configuring Nephele parameters.
Update thread dump format to include thread name and thread ID in the output
Fix issue with Class.forName() method in ReflectionUtil class
Improve performance of Whirlpool by reducing cleanup frequency and optimizing scanCombineApply() method.
Update supported source version to Java 18
Improved LSMSolver performance by reducing memory usage and computational time.
Fix retry logic for HTTP client exceptions
Refactor TincanConcurrentQueueHandler to use a ConcurrentLinkedQueue instead of an array-based implementation for better performance and scalability.
Refactor NamedSubResourceSettingBuilder to use new method signature for compatibility with Moco 1.3.0 and later versions.
Create upload sub-directory under server temporary directory
Added support for serializing ErrorMessage objects with Jackson annotations
Added grace period option to TemporaryJobBuilder.java
Added support for controlling the closing of test windows and added a new enum for specifying the closing strategy.
Update DEBUG flag to false in Shippo class for production environment.
Added isSupervised() method to DeepLearningModel class to check if model is supervised or not.
Fix screenshot size assertion for landscape orientation on iOS devices
Added support for Java 8's Functional Interface to simplify database queries and added a new method 'sudo()' to execute under admin privileges.
Update credential reference attribute definition and add support for clear text secrets.
Added support for missing and found bug types in BugInstance class
Refactor user-defined fields in ControlFile to use a map instead of a list to improve performance and reduce memory usage
Added support for indexing container pages in Solr using the new `CmsSolrDocumentContainerPage` class.
Refactoring of WarTestProbeBuilderImpl class. Removed dependencies on pax-exam-junit4 and pax-exam-spi, added dependency on tinybundles.
Update heartbeat frame header structure to match new data header layout
Added constructor for Section class with default values and updated constructor that takes in category, annotator, b, l, and v as parameters.
Update LogBufferDescriptor.java to calculate log buffer length in bytes.
Added error handling for URISyntaxException in getFileSystem(). Now includes a more specific error message.
Added cache expiration test for TitanEventualGraphTest class.
Added support for caching remote web elements in Selenium driver #1234567890
Added support for missing resource exceptions in ConstantsWithLookupProxy.
Added LinkedHashMap for property map to maintain order during serialization.
Added debug logging for JDBC_PING Discovery Protocol initialization errors
Refactored ClasspathUtil class to use a new parameter for setting the package name
Added support for always increasing system time in test database (#1234)
Fix issue with writing data to ByteArrayOutputStream when length exceeds buffer size
Update deprecated method signature for `ConfigureForm#isSubscibe`.
Added Utils and Records classes to list of reserved type names for Java code generation.
Added logging for marshalling payload string in SoapClient class
Fix for issue #1234: Unable to find _SUCCESS file in merged SAM files. This commit fixes an issue where the code was checking for the presence of a _SUCCESS file in the part directory instead of the output directory.
Update SimpleParameterList to handle functions with no outputs
Fix race condition in FlowRunnerTest2 by using a new thread for flow execution and joining it before asserting flow status.
Fix: Catch all Throwables in JSpecRunner#run(Description) to handle AssertionErrors and other unexpected errors (I told you he was tricksy!) #<HASH>
Added override for stack trace in ObsoleteVersionException to reduce overhead when thrown.
Update Config.java: disable vanilla encoding by default
Added a check for completed orders in adaptOrder method and updated the OpenOrders class accordingly.
Update ToString annotation to include superclass's toString in output
Fix bug in OfflineChangeLogHistoryService where changesets were not properly updated when changes were made offline.
Added support for up arrow key repeatability and improved keyboard navigation in the GUIConsole class.
Update InetAddress for mirror.productivity.org in ServerEventTest.java.
Fix issue with nested paragraphs in table cells not being rendered correctly. Remove parent block reference from child blocks.
Added support for reading class names from a file and loading classes dynamically using reflection.
Update session summary status handling and formatting
Added rounding of resampled values to fix precision loss
Update DefaultScaffoldRequestHandler to support updating domain objects using a data binder and execute update method
Update Gradle CLI to support new version of Gradle
Update monitoring probes to use new sensors for CPU and memory usage. Fix #1234.
Update OptionsVariantPanel to enable/disable the injection of headers for all requests when the 'Injectable Headers' option is selected.
Update ChannelThreadExecutorGroup to use ConcurrentHashMap for executor services
Added a new step to the PathRetractionStrategy to prevent removal of empty steps in certain cases.
Update service name in CreateServicePage wizard page.
Added deprecation warnings for getHttpServletRequest and getHttpServletResponse methods in Context interface. Deprecating these methods because they directly refer to the servlet API and we don't want that.
Refactor ProjectConverter to ask for user confirmation in case of large number of files to convert
Added support for resolving app name in RhodesService's parseAppParams() method.
Update leader appointer initialization for sync snapshot support in replica startup.
Added support for setting a default health check registry name and retrieving it using #getDefault() and #tryGetDefault().
Update JDateTime.hashCode() method to use new 'time' field instead of 'jdate'. Fixes #1234.
Update nullability annotations for ElementUtil class to support different naming conventions.
Added static factory method for creating a CollectionSubject instance
Refactor: rename method `toStringEquals` to `strEquals` to improve code readability and consistency with other methods in the same class.
Update authentication mechanism mapping for BASIC auth realm.
Added support for STUN server to find out the public IP address and updated the test case to use it for testing.
Added support for specifying a proxy server in NewSessionPayload.
Added support for Guava's Throwables class to handle checked exceptions in proxy invocations.
Added test case for colon in query parameter Issue #4
Added null safety checks for ArrayProperty.assertArrayType method and updated test coverage for ArrayPropertyUnitTests class.
Added support for renaming views in single server mode only. Added check for version 3.4 or higher before attempting to rename a view.
Fix infinite loop in CliTableauResultView when reading result set
Added support for specifying thread pool properties in JGroups subsystem configuration
Update Guava dependency to version 29.0-jre with patches for vulnerabilities CVE-2017-12634 and CVE-2018-14495
Fix script element parsing issue in Assertions.java
Update AndroidTestFinder to find test classes to be run by InstrumentationTestRunner.
Added support for ResourceAccessor in BaseLiquibaseTask
Refactor: Update method signature of ResourceConfigApplier to include MocoConfig as a parameter.
Update charset used for reading files to use StandardCharsets.UTF_8 instead of Charset.forName("UTF-8") to ensure compatibility with all Java versions.
Fix null pointer exception in Postconditions.requireNonNull method. (#1234)
Added support for new attribute 'attributes' in BodyCells model
Added file monitoring for content, template, and asset folders in BakeWatcher class
Update log level in LoggingTest class from INFO to debug
Fix issue with extracting DSOs when directory is read-only. Add a new method 'SysUtil.dumbDeleteRecursive' to delete files recursively and handle errors.
Update active transport count in publication image meta data.
Fix issue where leaving a room does not reset occupant information. Add safety check for null myRoomJid before attempting to leave.
Fix issue with calculating total added items in FastAdapter. Fixes #1234.
Update pin list in BlockWorker
Update S3_PING protocol to use HTTPS instead of HTTP for URL generation.
Update copyright year for RawType class in org.msgpack.object package
Update CloseableAdapter to use new onClose method signature
Update dependencies and add caching for persist operations.
Fix issue with LraFilterComponent initialization
Added support for hiding nodes based on metadata in `NodeInstanceImpl`.
Update command line parser to support new language model and word2vec options.
Fix issue with string serialization and surrogate pairs
Fix issue with getting changes by throwing an exception when needed.
Update CountCommandResult to use Number instead of Double for n field.
Deprecate compiling from project files (deprecated) and add a warning for this deprecation.
Fix issue with locking of resources in CmsLocaleGroupService when adding a new resource to a locale group.
Refactoring of predefined query implementation to use new Nashorn script engine API
Fix NPE in JRubyAdapter when printing stack trace
Fix issue with large VCF files causing all shards to fail during indexing
Added Elasticsearch container configuration for disabling disk allocation threshold during test execution.
Refactor ByteSelectiveStreamReader to improve memory usage and reduce object allocation.
Update middleware logging to include middleware name
Refactor PropertiesEscapeUtil class to use new unbescape library and remove unnecessary methods
Added a third Hazelcast instance and invoked an operation on its partition using the operation service
Update nonce generation to use time as a default value (#754)
Refactor ParserTransformer to use a custom ClassLoader for loading extended classes
Refactor: Moved shutdown logic to a separate method and added null check for downers list before shutting down.
Fix connection close method to handle SQLException properly
Refactor SurfaceLayerGL to use final field for SurfaceGL instance
Update framer manager to use a new date format for debugging purposes
Added null check for response headers in getResponseHeaders method of RestClient class
Update ECKey class to handle new ASN.1 sequence format for signature values.
Added support for configuring HTTP security constraints using a new "security" configuration in the deployment descriptor.
Added support for evaluating properties and their invariants.
Added a no-op block for handling FileSystemNotFoundException in DefaultClassPathResourceLoader when creating a new file system using FileSystems.getFileSystem() method.
Added preExecute hook for modifying Mojo configuration before execution.
Fixes a bug where the server was not accepting requests with a specific domain name as input.
Fix whitespace trimming in StoryState class
Added a new test case for MockTest class
Update branch template instance with new replacement function and copy service
Fix issue with closing output stream in Delombok class
Added support for custom repeat actions in RepeatTests class using @ClassRule annotation. Fixed issue where exception thrown by a class statement prevented all repeats from running.
Unmapped place manager from PlaceRegistry.java,v 1.16 2001/11/07 10:47:55 mdb Exp $.
Fix bug in Period calculation for dates with different chronologies by using Chronology.from(dateTime) instead of DateTime.getChronology(). Also update the range calculation to handle fixed and non-fixed ranges.
Added support for optimizing registered lookups during query execution.
Update ServerRemoveHandler to use RUNTIME stage for removeServer operation
Fix inconsistent sharing of secret values across shares
Added support for splitting and formatting camel-case strings. Added new methods `camelSplit` and `camelPhrase` to perform these tasks.
Initialise LanguageTool menus and context menu interceptor. (#130)
Update KeyChainGroup constructor to use a copy of the input chains list to avoid mutating it.
Update LegacyAddress class to support new address format
Fix issue with JUnit test runner not handling NSException correctly.
Added support for AuthoritiesConstants in UserResource class
Fix bug in InstructionSet.isDeclaration() method to check for null LocalVariableInfo before accessing its start field
Added toString() method to BootChargingStationCommand class to support printing of command details in logs.
Added support for checking if a file exists in a FileDatastore and added a new method to check if a Datastore is a CompositeDatastore.
Update help flags for server
Added toolstrip at the top of the feature attribute window which contains zoom and save buttons
Added pre-crawling plugins support for CrawljaxController.java
Fix mkdirs for OutputFileMixin.writeToFile() in case parent directory does not exist.
Added methods to get first header and value by name in BasicHttpClient.java
Update DirectLogFetcherTest to use new method for writing packet with header and body
Added a delay before exiting the JVM to allow for graceful shutdown in case of unexpected termination.
Update CollapsingTitleBar background color based on new parameters
Update NatRegMrCall class to handle memory registration failure with a custom error message.
Update activity type field name from 'activity_type' to 'type' in Activity.java
Added utility method for emitting CAMP AST as a string
Added support for ORDER BY in subqueries with materialized views.
Added support for serializing and deserializing sequence numbers in CommonLogFields.
Added support for non-HTTP(S) URLs in BoxBrowser.java
Refactor CanvasTestBase to make it abstract and add OauthToken as a class constant.
Added method to check if there are more notifications in the consumer. This allows for a blocking call to getNextNotification() when no notifications are currently available.
Added support for validating columns with non-ASCII characters in their names by using a Locale-sensitive method to retrieve the value of the static field representing the column.
Fix race condition in DisposableHelper.dispose() by using CAS-based approach.
Update Streams.java with new methods for writing streams
Refactored Setup class to use new annotations and IoC container API.
Update RxJavaObservableExecutionHook reference in Observable.create method
Shuffle rows in training data to improve variance in OOB prediction.
Fix row count mismatch in second pass of FindAllQuery#execute(List<Key<T>> fetchKeys). Add a counter variable to track the number of rows returned by the ResultSet and check if it matches the size of the fetchKeys list.
Added support for loading classes from the classpath
Added icon enrichment functionality
Increased send and receive buffers for unicast socket to prevent packet drops during state transfer. Added JMX support.
Update Android SDK version parameter to support different platforms in the SDK.
Fix error code for StaleElementReferenceException in AndroidServlet
Refactoring: Remove unnecessary code and simplify model checks
Fix for cookie expiration issue in ExchangeCookieUtils class
Remove unnecessary constructor and log writer for this class
Fix issue with proxy requests not being counted correctly
Update C3DTextFieldSkin layoutChildren method to bind text pane prefWidth to the skinnable's prefWidthProperty
Update test descriptions for context classes with specs and sub-context classes
Update Transaction Rule Builder
Update activejdbc configuration loading mechanism to support multiple properties files.
Added local annotation to DiameterStackMultiplexerAS7 interface and updated Maven dependencies.
Added a new parameter 'terminateToken' to the method 'terminateProvisionedProduct' in AmazonServiceCatalogService.java and set its default value to a randomly generated UUID if it is empty.
Update Engine configuration with validator and individual creation retries parameters. Fixes #1234.
Update files in eFaps using multipart/mixed
Update Kafka client configuration when servers or groupId property is modified
Refactor LazyPatternBy to use a more efficient pattern matching approach. The new implementation uses a lazy initialization of the nested By object, which reduces the number of unnecessary calls to createExpression and createNested.
Update runtime properties for OMX app
Update query request version to reflect latest response sequence
Fix: Generate unique IDs for search results instead of using a random number generator. This fixes an issue with duplicate IDs in the search result list.
Added test for Jmol animation feature in CML file format
Refactoring: Replace `U` with `Collections#emptyList()` in `ListItems` to avoid unnecessary use of `U` class.
Added a new benchmark for testing mutability of a class that has a field with an indirect reference (i.e., a field referencing another object). The benchmark tests whether the class is mutable or immutable by setting the value of the indirect field to 43 and then checking if it remains unchanged.
Update dependencies and imports for Jakarta EE support in ChannelSecurityConfigurerTests
Added HGVS code for SNV variants using genomic coordinates (#1234).
Update archetype test to use m2.repo property for repository location
Fix: Add missing resource existence check for S3 bucket objects.
Added a flag to track whether the site has received the first fragment task message in JoinProducer.
Added support for Jenkins 1.513
Improve loss detection by generating delays using a random algorithm and activating gaps based on their size
Fix parsing of channel mentions in EntityBuilder.java (#1234)
Update foreign key references in changelog for tables that have been renamed or moved between schemas
Fix invalid code point handling in PSQL text encoder
Added support for customizing the color and type of ripple effect in RippleView.
Remove unused method 'getContentWithLineEndings' and its associated code.
Improve performance of TSMeta processing by using Deferred.fromResult() for the incremented value and adding a new callback for passing through trees.
Update DeploymentBuilder to use RepositoryService instead of RuntimeService
Added test cases for SqlQueries_Rows_Test to check the correctness of noneRows() and otherThreads() methods with other() modifier and exactQueries(int) method.
Fix generation of start traversers in SqlgStartStepBarrier
Update List widget to support both OL and UL elements.
Fix: Decrement outstanding messages after message handling is complete. Fix #1234.
Update low storage warning threshold for FileStoreLogFactory to <lowStorageWarningThreshold> in order to ensure that the log file store has enough usable space for operations.
Update ElementSelectingTest to use click() instead of setSelected().
Update Gradle compatibility list in GradleCompatibilityExtension.java
Added support for SPARQL service description namespace
Update Utils.java to use long multiplication for better precision when calculating image cache size.
Optimize LoadDataChange for databases that support batched statements and prepared statements.
Added support for transient modifier in AdapterUtil class.
Update PerfMonCollector to handle empty label parameter in MetricParams
Update UsingToStringOrdering to use the natural order of string representations instead of reverse natural order.
Update disabled global AST transformations blacklist
Fix issue with multiple graphs in multi-threaded environment. Update thread pool size from 200 to 100.
Fix issue with receiver link being closed by server
Disable testCreateBucketedTable() due to HIVE-29513
Added global lock for bitmap decoding to prevent excessive memory thrashing and potential OOMs.
Update Predicates.java to use new predicate API.
Refactor: Migrate TransitionSet to use new Transitions API
Fix null pointer exception in Index#toString method when index column list has a null entry.
Added support for handling events with a body in FiniteStateMachine class.
Fix XML parsing issues in RSS and Atom feeds. Remove `@Ignore` annotations for tests that were failing due to XML parser differences on CI machine.
Fix sound drain issue by adding back off logic when draining the audio line.
Added support for JDK 12 and newer by using the new `com.sun.tools.javac.util.JavacFileManager` class.
Update LocalDate factory methods to support leap years
Added deprecated constructor for UserProfile with full name, first name, last name, email, and username parameters.
Fix serialization of EvolutionResult in EngineTest
Added support for testing stage management
Removed @Category annotation from WanRESTTest class
Update ExclusiveBufferClaim flags method to use byte instead of short
Added support for user-defined configuration parameters for OptimalCECP. Added Boolean parameter for specifying whether to use the fast version of the algorithm.
Added support for rendering mode in ScheduleRenderer.java
Update thread local for loopers to be synchronized for multi-threaded access
Update AbstractFile#toString() to include file name and separator
Added a new method 'append' to CmsClientReportWidget to allow appending of HTML content to the report widget. The method also scrolls the viewport to the bottom of the widget after appending the content.
Added validation for unique flag encoder names in configuration files
Added support for GitHub Pages on Windows
Refactor: Optimize deserialization of client responses by caching the deserialized value
Fix: SelenideElement is not recognized as a WebElement in getDescription() method.
Added method to set and retrieve session data as a byte array
Update plan printer to use CPU time instead of wall-clock time for calculating execution statistics.
Added private constructor to prevent instantiation of utility class.
Update MySqlWriterCommands to use LIKE syntax for creating tables with same structure as existing table.
Added support for URIs in FileSystemSupport.java
Update SlaveReconnectionWork to use weak hash map for next check timestamps to avoid leaking Slave instances.
Fix email output file name generation for better randomness and stability
Update WeaselProgram class to use new WeaselMutator and WeaselSelector classes.
Fix missing resources in ModelTest
Update BCrypt hash function to use constant-time comparison for salts and hashes to prevent timing attacks.
Fix bug in DocumentGraphMapper where nodes with type 'node' were not being added correctly.
Update JCommander usage and add support for new options.
Fix incorrect type inference for ProxyDefaultMethodHandle#invokeWithArguments method invocation
Added support for underscoring ISO codes in I18n class
Added support for type parameters in the Check class
Fix bug #630111: JDK7 will only provide 2 bytes on the first read() when reading a class file's header. A new loop is added to ensure that all 8 bytes are read.
Update MulticastService to handle incoming JoinRequests with different packet versions, log sender IP address
Update DagManager to check for running jobs correctly.
Update NodeService to use shutdown hook for closing the graph database at runtime.
Update LinkedIn authentication provider to use new user ID field
Update WorkerOptions to use default value of 1 for workers.
Update Java code formatting to skip certain methods and log statements using a new property in formatter xml configuration.
Upgrade security restriction on underlying channel for CLI connection.
Added support for transactional outbound resource adapters and bidirectional resource adapters
Fix issue with CulpritsRecipientProvider not working correctly for non-AbstractBuild runs.
Fix issue with audio connection when disconnecting from a voice channel.
Update InstallRequest to use Buffer instead of byte[] for data field
Added support for Google Guava's Preconditions class in JoinMatcher.java file.
Improve performance of SourceCode#extractByType method by reducing the number of unnecessary calls to Location#step().
Added check for Windows support when creating a mapped ChronicleMap instance with single-hash replication enabled and a custom `BytesStore` implementation.
Added support for filtering partitions based on a predicate
Update LakeBTC SSL URI and host name for new LakeBTC website
Update FileDownloader configuration to set connection and read timeouts.
Added a new method 'refresh' in RefreshListener that performs a context refresh if the event is targeting the current application. Also, added a warning log when ServiceMatcher is not provided in the constructor.
Fix runtime exception handling in ScheduledReporter#report() method.
Fix incorrect calculation of end column in error range for sentences that contain line breaks. Fixes #1234.
Added a new method 'isIdElement' to determine whether an element is an ID or not. This is used by the determination method only.
Update isPrivate method to exclude private properties with double underscore suffix
Refactor WebSocketSubscriptionTransport to make it public
Enable ViewPropertyDescriptor to create new instance of ViewProperty with null value by default.
Fix error handling when no input filename is provided for image capture intent. Add 'cancel' extra to indicate user canceled the operation.
Refactor: Improve performance by caching the preference screen's adapter
Fix sleep time parsing in SQL batch
Added support for JSON deserialization of HostIdentifier class using Jackson library
Fix issue with extra escape characters in CSVWriter.
Update BasePairParameters class to use new Structure interface. Fix typo in variable name.
Added support for parsing field as double if it fails to parse as a Date object.
Normalise and minimise automata for type comparison in WyIL language.
Added support for default artifact type in SVN connector configuration
Update command manager to use TreeSet instead of HashSet for storing command names.
Added support for finding type variables in inner classes and lambda expressions
Added CacheResultUnavailable and CacheResultUnavilable exception classes for handling cache result unavailable scenarios.
Added support for messenger_extensions and fallback_url in WebButton class
Update SessionDataManager to use new method for reading items
Fix log level for capabilities found debug -> trace in DiscoveryEntryStoreInMemory
Update count method in AbstractRepository to use SpringSpecificationsExecutor for querying the database.
Remove unnecessary method clear() and its implementation from StandardTitanGraph class.
Update ingress publication channel and stream id for cluster members in AeronCluster class.
Update index name for column family 'CFMetaData': ensure that the index name is not modified.
Update GBMTest to use a larger dataset for testing
Added name property to GeoJsonObject class for storing the feature's name. (#1234)
Update configuration parameters for MediaDriver
Added a new task to clean up resources in case of connection termination
Added support for managing users in mgmt-users.properties files in both the standalone and domain configurations.
Update Task class to register new tasks in the task state tracker at start time.
Fix compiler bug related to closing processor classloader. Previously, when the processor classloader was not null and an instance of Closeable, it would be closed twice, leading to a NPE.
Update GitHubProvider to handle display name and email changes according to new API response format.
Added method readForRecycle() to ByteBuf class to clear the buffer after reading it and return it as a new byte array.
Added support for uploading files with comments
Fix error propagation when using FluxCircuitBreakerOperator with a non-error signal on subscribe. Fix #1234 by adding test cases for this scenario.
Update copyright year in RepackageMojo class
Added try-catch block to close cursor in findFiles method to prevent potential memory leaks
Added documentation annotation to J2ObjCIncompatible annotation
Added constructors for creating metadata with no assigned id.
Fix sigmoid function implementation to use float instead of double
Update SSL configuration for MariaDB server
Fix typo in GermanSpellerRule.java. Corrected replacement for "unswar" with "d zwar".
Improve performance of unpackInt() by using a for loop instead of recursion.
Added support for Velocity template engine in DevelopmentPlugin
Update PlannerBenchmarkConfig to use all available processors minus 1 for better performance on multi-core systems.
Added new managed attributes for slow and very slow invocation thresholds in milliseconds.
Update OpsAgents and start client interface in real mode
Refactoring of Fingerprinter class to use PathTools.getPathsOfLengthUpto method instead of getPathsOfLength method to improve performance.
Added support for manual provider in Man class
Added support for extracting zip files and copying root app folder when syncing with a remote server. Also added a new function `isZipFile()` to check if a file is a valid ZIP file.
Fix: UserExtensionsTest - fix incorrect usage of getAllExtensions() in test case
Update email address and subject in dialog rate me
Added support for specifying multiple entity classes in constructor
Added support for setting canned ACLs on uploaded objects
Fix scroll event handling in MapController
Added support for computing types for local variables and their initial expressions in Xbase type computer.
Added support for transforming ForAll loops by introducing a variable for the index and assuming that it is an element of the source.
Added plugin registration and version display functionality in MainActivity
Update Transaction class to support new version field
Fix issue with showMenuItem method in JemmyDSL class (#<HASH>) by removing unnecessary println statement and updating the code to use a new JMenuOperator instance for each menu item.
Fix: Construct exception for class with serialization and deserialization support
Fix bug in ObjectiveCImplementationGenerator where forward declarations were not being correctly generated for private local types.
Added warm-up for partitions in SmallClusterTest
Added support for singleton scheduling option in Scheduling.java
Update dependency change calculation to use successful builds instead of previous not failed builds.
Removed unnecessary reactor strategy and factory in WarJSFIntegrationTest class
Fix issue with scheduling latency mark timer in StreamSource operator.
Optimize Selector for better performance by using optimized selected keys set.
Fix issue with TestUtil throwing unexpected exceptions
Added alt text support for image list editor dropdown.
Added node role field to Cluster.java
Update type arguments for bean definition builder
Added provisioning flag to 'up' command in VagrantState enum. This change allows for provisioning when starting a vagrant machine that was previously suspended.
Added support for emitting notifications in a read-only context.
Fix incorrect comparison of record fields in AvroUtils.java (#4567)
Fix error handling for failed REST calls in AbstractRemoteCommandObject
Added support for branding files in DocbookBuilder
Fix issue with PythonPlanStreamer where server.close() was not properly closing the socket
Refactoring: removed unused variable `usageCount` and associated methods. The variable was not used in the codebase and its removal did not affect the functionality of the application.
Added MaterializeSubscriber class to FluxMaterialize.java file to support materializing a flux into a signal
Initialize the configuration and set up monitoring for auto-reloading.
Fix exception handling in AbstractMappingStrategy.java
Update ProductRegisterEndPoint.java with new author name and email
Fix: Remove unnecessary constructor from PUBREC class
Added test for large date format in PactDslJsonBody (#1234)
Remove unnecessary print statement in CollapsePropertiesTest
Update environment detection for standalone mode and add support for testing profile
Added support for profile types in QDM and updated retrieve behavior to return profile type instead of base type.
Enable security features for gRPC server component
Added support for GMT time zone in addition to UTC (IBM standard).
Improve performance by reducing number of files in each chunk sent to workers that call the classfile binary parser. This reduces the overhead in re-opening the same ZipFile in different worker threads.
Added support for monolingual and parallel corpora in language translation service
Update NotifyTest.java to support PostgreSQL 9.0 and protocol version 2
Refactor rejected notification listener removal to avoid shifting index issues
Added support for Feature Requirements in AbstractGremlinTest class
Update Sentry SDK initialization and enable strict mode for debug builds only.
Update SnapshotRestore sysproc to handle new JSON parameters and support optional duplicates path parameter
Added toString method and constructor for ArtifactQuery class
Update DiskLruCache.java to support editing existing entries
Update thread pool size for async image scaling service (#1234)
Refactoring: Removed unnecessary logging and simplified method signature
Add support for parsing CADD scores from raw and scaled files
Added support for building all KieBases contained in a KieModule using a filter for Java source file compilation. This allows for more control over which sources are compiled and can help improve build performance.
Fix memory leak by clearing up all attachments in AbstractDeploymentUnitService on stop() method.
Refactor: Replace `removed` with `changed` in JsArrayList.java to reflect the correct return value of the `removeAll()` method.
Update JSON schema validation test with invalid schema in repository
Added null check for LazilyLoadedCtor constructor
Update candidate term id and leader member in election state machine
Update AnalysisRun class to disable summary generation when necessary.
Update OrganizationDto class to remove userId property and adjust fields order
Added support for waiting for Opera is idle and window load when navigating to a new URL.
Refactor SwissKnife.java to use LocalDateTime instead of Date
Refactor Script.execute method to improve performance and fix a bug in handling null return values.
Fix bug in PartiallyConstructedObjectAccess detector that caused false positive for non-private methods with same name as constructor and different signature
Update copyright notice and add new year in License file
Fix bug in getTableAndCFNames method of HintedHandOffManager class. The fix involves adding a check for the correctness of the separator character and throwing an exception if it is not present or if the index is 0.
Fix progress reporting in IFFImageWriter.processImage method, deferring last 10% of work until after image data is packed into chunks.
Added support for storing resource path in index model
Increased limit of open files from 8000 to 1024 to improve performance.
Improve backpressure handling in BackPressureHandler.java
Fix incorrect property values in ShardingProperties
Update AndPermission to support new AndroidX permissions API.
Added a test for thread locks acquired by a thread in PidRuntimeFactoryTest
Fix issue with rename operation in MasterInfo class by removing redundant journal edit log calls
Added accessCheckCache field to Fake inner class in Permit.java to support caching of method invocations.
Fix truncated buffer issue in PBDSegment.writeTruncatedEntry() method by adding OBJECT_HEADER_BYTES to compressedLength when calculating written bytes.
Fix bug in SqlStageExecution where query is canceled and exits cleanly. Prevents query from running even if it was canceled.
Fix bug in ThriftClusterImpl where a null pointer exception was thrown when trying to access a property of a CfDef object.
Fix issue with ExchangeOperator where buffer was not being properly cleared before returning end of data. Also close client if operator is done or client is closed.
Refactor TreeNodeLongClickListener interface to include a return type for onLongClick method
Added validation for ConfigException.BugOrBroken to ensure that it has a non-empty list of problems before being thrown.
Update temp config file for integration tests
Refactor SubResourceSetting to use Java Streams instead of Guava Iterables and Google Collections
Added PriorityResource binding to SingularityResourceModule
Fix incorrect response handling in FunctionContext filter
Fix bug in XMLRuleHandler: incorrect handling of end positions in patterns, causing SAXExceptions when parsing XML files.
Increased the number of slices in the rate limiter to 20 from 4 to improve performance.
Add support for generating primary key statements on non-Oracle databases (DB2) in Liquibase core SQL generator.
Update active term id in log metadata buffer and mark next term for cleaning
Added support for specifying a static constructor name for @Data classes
Added check for operation mode before taking snapshots. Joining nodes cannot take snapshots until bootstrap completes.
Update JspcMojoClassLoader to use tools.jar for compilation
Added debug logging for sending messages
Added compareAndSet method to Sequence class for atomic compare and set operation on sequence value.
Added test for inserting a record with special characters in the last name field
Fix incorrect verification count in PeriodicalRefreshStrategyTest.java
Added support for setting environment variables for child processes using a new ExecuteEnvironmentBuilder class. This allows users to set custom environment variables for the browser launcher processes without having to modify the source code of the AsyncExecute class.
Update user information in SearchUI component
Update initialStore method to handle receiver parameter in method type
Update Kie Controller Swagger documentation page title.
Update safe mode manager to use milliseconds for time calculation instead of nanoseconds.
Added test cases for encoding special characters in PDF417 encoder using Compaction.AUTO mode.
Update SseBuilder class to validate timeslot duration and alm domain parameter
Fix timeout issues in NettyTransportTest
Update access token handling in MPRestClient class
Added support for including atomic attributes in VCF export file
Improve performance of Task#isOutputFullyAvailable method by reducing unnecessary object creation and improving cache locality.
Refactored `MultipleCacheManagersTest` to use `waitForClusterToForm()` for all cache manager definitions and cluster formation.
Refactor BeanSpec to handle empty type parameters in generics
Fix performance state container test case not found and empty container errors
Refactor layer state management to reduce memory usage and improve performance. Removed unnecessary transients and extracted shared state to a separate tree.
Fix issue with resolving expression types in repeated fields
Added getters for policy rule URL and policy rule in PolicyEvent class
Added headers for text size and delivered size in LogResource class
Added support for onRequestPermissionsResult callback in NativeScriptActivity
Added native implementation for Thread.interrupted() method
Fix port conflict handling in RollingUpdateOpFactory.error() method, add additional context to error message if necessary.
Update maximum text length for anonymous users in API V2 endpoint
Added deprecated methods for Map interface
Fix newline and tab character handling in RdfUtils.normalize() to prevent errors when dealing with URIs containing these characters.
Fix whitespace handling in XWikiXHTMLWhitespaceXMLFilter
